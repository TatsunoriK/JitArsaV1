"""
update_database.py
==================
Auto-update pipeline สำหรับ JitArsa chatbot
รัน: python update_database.py

Flow:
  1. Scrape jitarsabank.com (Playwright)
  2. Clean + merge เข้า jitarsa.json
  3. POST /reload-data → main.py rebuild vector ใหม่
"""

import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd
from playwright.async_api import async_playwright

# ─── CONFIG ──────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
DATA_PATH     = BASE_DIR / "data" / "jitarsa.json"
BACKUP_DIR    = BASE_DIR / "data" / "backups"
MAX_PAGES     = 10          # จำนวนหน้าที่ scrape จาก jitarsabank
DELAY_MS      = 400         # หน่วงระหว่าง request (ms)
RELOAD_URL    = "http://127.0.0.1:8000/reload-data"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
# ─────────────────────────────────────────────────────────


# ══════════════════════════════════════════════
# 1) SCRAPE LIST
# ══════════════════════════════════════════════
async def scrape_list(browser) -> list[dict]:
    """ดึงรายการ URL จากหน้า listing"""
    ctx  = await browser.new_context(user_agent=UA)
    page = await ctx.new_page()
    urls = []
    prev = set()

    for pg in range(1, MAX_PAGES + 1):
        url = f"https://www.jitarsabank.com/?filter%5Bjob_status%5D=publish&page={pg}"
        try:
            await page.goto(url, wait_until="networkidle", timeout=30_000)
            await page.wait_for_selector("article.card", timeout=15_000)
        except:
            print(f"  ⚠️  ไม่มีข้อมูลหน้า {pg} — หยุด")
            break

        cards = await page.query_selector_all("article.card")
        cur   = set()
        for c in cards:
            el = await c.query_selector("a.card-link")
            if el:
                href = await el.get_attribute("href") or ""
                full = ("https://www.jitarsabank.com" + href) if href.startswith("/") else href
                cur.add(full)

        if cur == prev:
            print(f"  ⚠️  หน้า {pg} ซ้ำ — หยุด")
            break
        prev = cur
        urls += list(cur)
        print(f"  📄 หน้า {pg}: +{len(cur)} URL (รวม {len(urls)})")

    await ctx.close()
    return [{"url": u} for u in dict.fromkeys(urls)]  # dedup ลำดับ


# ══════════════════════════════════════════════
# 2) SCRAPE DETAIL
# ══════════════════════════════════════════════
async def scrape_detail(page, url: str) -> dict | None:
    try:
        await page.goto(url, wait_until="networkidle", timeout=30_000)
        await page.wait_for_selector("article, main, #app", timeout=15_000)
    except:
        return None

    try:
        el = (
            await page.query_selector("main") or
            await page.query_selector("#app")  or
            await page.query_selector("body")
        )
        raw = (await el.inner_text()).strip() if el else ""

        # ตัด nav + footer
        raw = re.sub(r"เกี่ยวกับเรา.*?สมัครสมาชิกเป็นอาสา\s*", "", raw, flags=re.DOTALL)
        raw = re.sub(r"ชวนเพื่อนไปงานอาสานี้.*$", "", raw, flags=re.DOTALL).strip()

        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        if not lines:
            return None

        # --- parse fields ---
        title   = lines[0]
        org     = ""
        date    = ""
        loc     = ""
        cost    = ""
        phone   = ""
        email_  = ""
        seats_r = 0
        seats_t = 0

        for line in lines:
            ll = line.lower()
            if "องค์กร" in ll or "โดย" in ll:
                org = line.split(":")[-1].strip() if ":" in line else line
            elif re.search(r"\d{1,2}:\d{2}", line) and ("ม.ค\.|ก.พ\.|มี.ค\.|เม.ย\.|พ.ค\.|มิ.ย\.|ก.ค\.|ส.ค\.|ก.ย\.|ต.ค\.|พ.ย\.|ธ.ค\.)", line):
                date = line
            elif "ค่าใช้จ่าย" in ll or "ฟรี" in ll or "ไม่มีค่า" in ll:
                cost = "ไม่เสียค่าใช้จ่าย" if any(k in ll for k in ["ฟรี","ไม่มีค่า","ไม่เสีย"]) else "มีค่าใช้จ่าย"
            elif "ที่นั่งสมัครแล้ว" in ll:
                nums = re.findall(r"\d+", line)
                if nums: seats_r = int(nums[0])
            elif "ที่นั่งทั้งหมด" in ll:
                nums = re.findall(r"\d+", line)
                if nums: seats_t = int(nums[0])
            elif re.match(r"0[0-9]{8,9}$", line.strip()):
                phone = line.strip()
            elif "@" in line and "." in line:
                email_ = line.strip()
            elif any(k in ll for k in ["กรุงเทพ","เชียงใหม่","สงขลา","ทำที่บ้าน","ออนไลน์"]) and not loc:
                loc = line

        # เนื้อหา = ทุกอย่างหลังชื่อ
        detail = "\n".join(lines[1:])

        return {
            "ชื่อกิจกรรม":         title,
            "ชื่อองค์กร":           org,
            "วันที่-เวลา":          date,
            "สถานที่":              loc,
            "มีค่าใช้จ่าย":        cost != "ไม่เสียค่าใช้จ่าย",
            "ที่นั่งสมัครแล้ว":    seats_r,
            "ที่นั่งทั้งหมด":      seats_t,
            "เบอร์ติดต่อ":         phone,
            "อีเมล":               email_,
            "เว็บไซต์":            "",
            "รายละเอียด":          detail,
            "url":                  url,
        }
    except Exception as e:
        print(f"  ⚠️  parse error {url}: {e}")
        return None


# ══════════════════════════════════════════════
# 3) SCRAPE ALL
# ══════════════════════════════════════════════
async def scrape_all() -> list[dict]:
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"]
        )

        print("📋 ดึงรายการ URL...")
        items = await scrape_list(browser)
        print(f"  → พบ {len(items)} URL\n")

        print("🔍 ดึงรายละเอียดแต่ละงาน...")
        ctx  = await browser.new_context(user_agent=UA)
        page = await ctx.new_page()

        results = []
        for i, item in enumerate(items, 1):
            url = item["url"]
            print(f"  [{i}/{len(items)}] {url}")
            data = await scrape_detail(page, url)
            if data:
                results.append(data)
            await page.wait_for_timeout(DELAY_MS)

        await ctx.close()
        await browser.close()

    print(f"\n✅ scrape เสร็จ: {len(results)} รายการ")
    return results


# ══════════════════════════════════════════════
# 4) CLEAN
# ══════════════════════════════════════════════
def clean_data(records: list[dict]) -> list[dict]:
    df = pd.DataFrame(records)

    # ลบ title ว่าง
    df = df[df["ชื่อกิจกรรม"].str.strip() != ""]

    # dedup ตาม url
    df = df.drop_duplicates(subset=["url"])

    # fillna
    df = df.fillna("ไม่ระบุ")

    # เพิ่ม id ใหม่
    df = df.reset_index(drop=True)
    df.insert(0, "id", range(len(df)))

    return df.to_dict(orient="records")


# ══════════════════════════════════════════════
# 5) MERGE กับ JSON เดิม
# ══════════════════════════════════════════════
def merge_with_existing(new_records: list[dict]) -> tuple[list[dict], int]:
    existing = []
    if DATA_PATH.exists():
        with open(DATA_PATH, encoding="utf-8") as f:
            existing = json.load(f)

    existing_urls = {r.get("url", "") for r in existing}
    added = [r for r in new_records if r.get("url", "") not in existing_urls]

    # รวม: ของใหม่อยู่ข้างหน้า + ของเก่า
    merged = new_records  # replace ทั้งหมด (ข้อมูลใหม่ล่าสุด)

    # re-index id
    for i, r in enumerate(merged):
        r["id"] = i

    return merged, len(added)


# ══════════════════════════════════════════════
# 6) SAVE + BACKUP
# ══════════════════════════════════════════════
def save(records: list[dict]):
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    # backup ไฟล์เก่า
    if DATA_PATH.exists():
        ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = BACKUP_DIR / f"jitarsa_{ts}.json"
        backup.write_text(DATA_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"  💾 backup → {backup.name}")

    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"  ✅ บันทึก {len(records)} รายการ → {DATA_PATH}")


# ══════════════════════════════════════════════
# 7) RELOAD VECTOR (แจ้ง main.py)
# ══════════════════════════════════════════════
def reload_vector():
    try:
        r = httpx.post(RELOAD_URL, timeout=120)
        if r.status_code == 200:
            print("  🔄 main.py rebuild vector เรียบร้อย")
        else:
            print(f"  ⚠️  reload ได้ status {r.status_code}")
    except Exception as e:
        print(f"  ⚠️  ไม่สามารถ reload ได้: {e}")
        print("     (ถ้า server ยังไม่รัน ไม่เป็นไร — reload ครั้งถัดไปเมื่อ restart)")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
async def main():
    print("=" * 50)
    print(f"🚀 JitArsa DB Update  —  {datetime.now():%Y-%m-%d %H:%M}")
    print("=" * 50)

    raw     = await scrape_all()
    cleaned = clean_data(raw)
    merged, added = merge_with_existing(cleaned)

    print(f"\n📊 สรุป: ทั้งหมด {len(merged)} รายการ (ใหม่ {added})")

    print("\n💾 บันทึกไฟล์...")
    save(merged)

    print("\n🔄 แจ้ง server reload vector...")
    reload_vector()

    print("\n✅ อัพเดตเสร็จสมบูรณ์!")


if __name__ == "__main__":
    asyncio.run(main())
