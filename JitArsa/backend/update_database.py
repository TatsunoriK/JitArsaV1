"""
update_database.py
==================
Auto-update pipeline สำหรับ JitArsa chatbot
รัน: python update_database.py

Flow:
  1. Scrape jitarsabank.com ด้วย Playwright (headless browser)
  2. Clean + merge เข้า jitarsa.json
  3. POST /reload-data → main.py rebuild FAISS vector ใหม่
"""

import asyncio    # รัน async functions (Playwright ต้องใช้ async)
import json       # อ่าน/เขียน JSON dataset
import os
import re         # regex parse ข้อมูลจาก HTML text
import sys
from datetime import datetime
from pathlib import Path  # จัดการ path แบบ OOP สะดวกกว่า os.path

import httpx                              # ส่ง HTTP request ไป /reload-data
import pandas as pd                       # clean + dedup ข้อมูลหลัง scrape
from playwright.async_api import async_playwright  # headless browser สำหรับ scrape

# ─── CONFIG ──────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent                 # โฟลเดอร์ที่ script นี้อยู่
DATA_PATH  = BASE_DIR / "data" / "jitarsa.json"   # path ไฟล์ dataset หลัก
BACKUP_DIR = BASE_DIR / "data" / "backups"        # โฟลเดอร์เก็บ backup ก่อน overwrite
MAX_PAGES  = 10    # จำนวนหน้า listing สูงสุดที่จะ scrape (กัน infinite loop)
DELAY_MS   = 400   # หน่วงระหว่าง request ป้องกัน rate limit และลด load ของ server
RELOAD_URL = "http://127.0.0.1:8000/reload-data"  # endpoint ของ main.py
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"  # User-Agent กัน bot detection
# ─────────────────────────────────────────────────────────


# ══════════════════════════════════════════════
# 1) SCRAPE LIST — ดึง URL ของทุกงานจากหน้า listing
# ══════════════════════════════════════════════
async def scrape_list(browser) -> list[dict]:
    """
    วนดึง URL จากหน้า listing ของ jitarsabank.com
    หยุดเมื่อ: ไม่มีข้อมูลหน้าใหม่ หรือหน้าซ้ำกับหน้าก่อน
    คืน list ของ {"url": "..."} โดยไม่มี URL ซ้ำ
    """
    # สร้าง browser context ใหม่พร้อม User-Agent ที่กำหนด
    ctx  = await browser.new_context(user_agent=UA)
    page = await ctx.new_page()
    urls = []
    prev = set()  # เก็บ URL ของหน้าก่อนหน้า สำหรับตรวจ duplicate

    for pg in range(1, MAX_PAGES + 1):
        url = f"https://www.jitarsabank.com/?filter%5Bjob_status%5D=publish&page={pg}"
        try:
            # wait_until="networkidle" รอให้ network request หยุดก่อน (SPA โหลดข้อมูลผ่าน JS)
            await page.goto(url, wait_until="networkidle", timeout=30_000)
            # รอให้ article.card ปรากฏ — ถ้าไม่มี = หน้าว่าง
            await page.wait_for_selector("article.card", timeout=15_000)
        except:
            print(f"  ⚠️  ไม่มีข้อมูลหน้า {pg} — หยุด")
            break

        # ดึง link ของทุก card ในหน้านี้
        cards = await page.query_selector_all("article.card")
        cur   = set()
        for c in cards:
            el = await c.query_selector("a.card-link")
            if el:
                href = await el.get_attribute("href") or ""
                # href อาจเป็น relative path (/jobs/123) หรือ absolute (https://...)
                full = ("https://www.jitarsabank.com" + href) if href.startswith("/") else href
                cur.add(full)

        # ถ้า URL ในหน้านี้เหมือนหน้าก่อนเป๊ะ = pagination ไม่มีหน้าใหม่แล้ว
        if cur == prev:
            print(f"  ⚠️  หน้า {pg} ซ้ำ — หยุด")
            break
        prev = cur
        urls += list(cur)
        print(f"  📄 หน้า {pg}: +{len(cur)} URL (รวม {len(urls)})")

    await ctx.close()
    # dict.fromkeys() ลบ duplicate แต่รักษาลำดับไว้ (Python 3.7+)
    return [{"url": u} for u in dict.fromkeys(urls)]


# ══════════════════════════════════════════════
# 2) SCRAPE DETAIL — ดึงข้อมูลละเอียดของแต่ละงาน
# ══════════════════════════════════════════════
async def scrape_detail(page, url: str) -> dict | None:
    """
    เปิดหน้างานแต่ละหน้า แล้ว extract ข้อมูลออกมา
    ใช้ text-based parsing (inner_text) แทน HTML parsing
    เพราะ layout อาจเปลี่ยนได้ แต่ text content คงที่กว่า

    โครงสร้าง inner_text ของแต่ละหน้างาน (สม่ำเสมอ):
      บรรทัด 0 : ชื่อกิจกรรม  ← lines[0]
      บรรทัด 1 : ชื่อองค์กร   ← lines[1]  (FIX: ดึงตำแหน่งแทน keyword scan)
      บรรทัด 2 : วันที่-เวลา  ← lines[2]  (FIX: ดึงตำแหน่งแทน regex ที่พลาด)
      บรรทัด 3 : สถานที่ (อาจมีคำว่า "(แผนที่)" ต่อท้าย)  ← lines[3]
      ...        เนื้อหาอื่นๆ

    คืน dict ของข้อมูลงาน หรือ None ถ้าโหลด/parse ไม่ได้
    """
    try:
        await page.goto(url, wait_until="networkidle", timeout=30_000)
        # รอ container หลักก่อน parse
        await page.wait_for_selector("article, main, #app", timeout=15_000)
    except:
        return None  # โหลดหน้าไม่ได้ → ข้ามไป

    try:
        # ลอง selector ตามลำดับความเฉพาะเจาะจง: main > #app > body
        el = (
            await page.query_selector("main") or
            await page.query_selector("#app")  or
            await page.query_selector("body")
        )
        raw = (await el.inner_text()).strip() if el else ""

        # ตัด navigation และ footer ออก (ไม่ใช่ข้อมูลงาน)
        raw = re.sub(r"เกี่ยวกับเรา.*?สมัครสมาชิกเป็นอาสา\s*", "", raw, flags=re.DOTALL)
        raw = re.sub(r"ชวนเพื่อนไปงานอาสานี้.*$", "", raw, flags=re.DOTALL).strip()

        # แบ่งเป็น lines และลบ empty lines
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        if not lines:
            return None

        # ─── FIX: ดึง org / date / location จากตำแหน่งคงที่ใน detail block ───
        # หน้างานของ jitarsabank มีโครงสร้าง inner_text ที่สม่ำเสมอ:
        # [0] ชื่องาน → title
        # [1] ชื่อองค์กร → org   (เดิม: scan หา "โดย" → ชนกับบรรทัดเกียรติบัตร)
        # [2] วันที่-เวลา → date  (เดิม: regex เดือนย่อพลาดบางรูปแบบ)
        # [3] สถานที่ (อาจมี "(แผนที่)" ต่อท้าย) → loc
        # การดึงตามตำแหน่งแม่นกว่า keyword scan สำหรับโครงสร้างที่แน่นอน

        title = lines[0]

        # org = บรรทัดที่ 1 ถ้ามีอยู่ และไม่ได้เป็นวันที่หรือ keyword อื่น
        # fallback ไปใช้ keyword scan เดิม ถ้าโครงสร้างผิดปกติ
        org = ""
        if len(lines) > 1:
            candidate = lines[1]
            # ถ้าบรรทัดที่ 1 ไม่ใช่วันที่ (ไม่มี HH:MM) = org
            if not re.search(r"\d{1,2}:\d{2}", candidate):
                org = candidate
            else:
                # โครงสร้างผิดปกติ → fallback keyword scan (เฉพาะ "องค์กร:" เท่านั้น ไม่ใช้ "โดย")
                for line in lines:
                    if "องค์กร" in line and ":" in line:
                        org = line.split(":")[-1].strip()
                        break

        # date = บรรทัดแรกที่มี HH:MM + เดือนภาษาไทย (ค้นหาจาก lines[1] เป็นต้นไป)
        # FIX: เพิ่ม pattern เดือนเต็ม (พฤษภาคม ฯลฯ) นอกจากเดือนย่อ
        date = ""
        _month_pattern = (
            r"(ม\.ค\.|ก\.พ\.|มี\.ค\.|เม\.ย\.|พ\.ค\.|มิ\.ย\.|ก\.ค\.|ส\.ค\.|"
            r"ก\.ย\.|ต\.ค\.|พ\.ย\.|ธ\.ค\.|"
            r"มกราคม|กุมภาพันธ์|มีนาคม|เมษายน|พฤษภาคม|มิถุนายน|"
            r"กรกฎาคม|สิงหาคม|กันยายน|ตุลาคม|พฤศจิกายน|ธันวาคม)"
        )
        for line in lines[1:]:
            if re.search(r"\d{1,2}:\d{2}", line) and re.search(_month_pattern, line):
                date = line
                break  # เจอบรรทัดแรกที่ตรงเงื่อนไข → หยุด

        # loc = บรรทัดที่มีคำว่า "(แผนที่)" ซึ่งเป็น pattern เฉพาะของ jitarsabank
        # หรือ fallback เป็นบรรทัดที่ 3 ถ้าไม่เจอ "(แผนที่)"
        loc = ""
        for line in lines[1:]:
            if "(แผนที่)" in line:
                # ลบ " (แผนที่)" ออก เก็บแค่ชื่อสถานที่
                loc = line.replace("(แผนที่)", "").strip()
                break
        if not loc and len(lines) > 3:
            # fallback: บรรทัดที่ 3 (หลัง title/org/date) มักเป็นสถานที่
            loc = lines[3]
        # ─────────────────────────────────────────────────────────────────

        # fields อื่นๆ ยังใช้ keyword scan เหมือนเดิม (ไม่มีตำแหน่งคงที่)
        cost    = ""
        phone   = ""
        email_  = ""
        seats_r = 0  # ที่นั่งที่สมัครแล้ว
        seats_t = 0  # ที่นั่งทั้งหมด

        for line in lines:
            ll = line.lower()
            if "ค่าใช้จ่าย" in ll or "ฟรี" in ll or "ไม่มีค่า" in ll:
                cost = "ไม่เสียค่าใช้จ่าย" if any(k in ll for k in ["ฟรี","ไม่มีค่า","ไม่เสีย"]) else "มีค่าใช้จ่าย"
            elif "ที่นั่งสมัครแล้ว" in ll:
                nums = re.findall(r"\d+", line)
                if nums: seats_r = int(nums[0])
            elif "ที่นั่งทั้งหมด" in ll:
                nums = re.findall(r"\d+", line)
                if nums: seats_t = int(nums[0])
            elif re.match(r"0[0-9]{8,9}$", line.strip()):
                # เบอร์โทรไทย: ขึ้นต้นด้วย 0 ตามด้วยเลข 8-9 หลัก
                phone = line.strip()
            elif "@" in line and "." in line and "อีเมล" not in line:
                # email: มีทั้ง @ และ . ในบรรทัดเดียว
                # ยกเว้นบรรทัดที่ขึ้นต้นด้วย "อีเมล:" (จะ parse แยก)
                email_ = line.strip()
            elif line.startswith("อีเมล") and ":" in line:
                # FIX: บาง record มี "อีเมล: xxx@xxx" → ดึงเฉพาะ address
                email_ = line.split(":", 1)[-1].strip()

        # รายละเอียดทั้งหมด = ทุก line หลังชื่อกิจกรรม
        detail = "\n".join(lines[1:])

        # คืน dict ที่มี field ตรงกับ schema ของ jitarsa.json
        return {
            "ชื่อกิจกรรม":      title,
            "ชื่อองค์กร":        org,
            "วันที่-เวลา":       date,
            "สถานที่":           loc,
            "มีค่าใช้จ่าย":     cost != "ไม่เสียค่าใช้จ่าย",  # bool: True=มีค่าใช้จ่าย
            "ที่นั่งสมัครแล้ว": seats_r,
            "ที่นั่งทั้งหมด":   seats_t,
            "เบอร์ติดต่อ":      phone,
            "อีเมล":            email_,
            "เว็บไซต์":         "",
            "รายละเอียด":       detail,
            "url":              url,
        }
    except Exception as e:
        print(f"  ⚠️  parse error {url}: {e}")
        return None


# ══════════════════════════════════════════════
# 3) SCRAPE ALL — orchestrate ทั้งหมด
# ══════════════════════════════════════════════
async def scrape_all() -> list[dict]:
    """
    เปิด Playwright browser แล้วรัน scrape 2 ขั้นตอน:
    1. scrape_list: ดึง URL ทั้งหมดจาก listing
    2. scrape_detail: วนเปิดทีละหน้าดึงข้อมูลละเอียด
    ใช้ browser เดียวกัน แต่แยก context (step 1) และ page (step 2)
    """
    async with async_playwright() as pw:
        # headless=True ไม่เปิด UI browser
        # --no-sandbox, --disable-dev-shm-usage แก้ปัญหาบน Linux/Docker
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
            # หน่วง DELAY_MS ms ระหว่าง request กัน rate limit
            await page.wait_for_timeout(DELAY_MS)

        await ctx.close()
        await browser.close()  # ปิด browser เมื่อเสร็จ

    print(f"\n✅ scrape เสร็จ: {len(results)} รายการ")
    return results


# ══════════════════════════════════════════════
# 4) CLEAN — ทำความสะอาดข้อมูลหลัง scrape
# ══════════════════════════════════════════════
def clean_data(records: list[dict]) -> list[dict]:
    """
    ทำความสะอาดข้อมูลที่ scrape มา:
    1. ลบแถวที่ชื่อกิจกรรมว่าง (scrape ผิดพลาด หรือหน้า redirect)
    2. dedup ตาม URL (กัน scrape ซ้ำ)
    3. fillna สำหรับ field ที่อาจเป็น NaN
    4. reset index + เพิ่ม id field ใหม่
    """
    df = pd.DataFrame(records)
    df = df[df["ชื่อกิจกรรม"].str.strip() != ""]  # ลบ title ว่าง
    df = df.drop_duplicates(subset=["url"])         # dedup ตาม URL
    df = df.fillna("ไม่ระบุ")
    df = df.reset_index(drop=True)
    df.insert(0, "id", range(len(df)))  # id เริ่มจาก 0 ต่อเนื่อง
    return df.to_dict(orient="records")


# ══════════════════════════════════════════════
# 5) MERGE — รวมกับข้อมูลเดิม
# ══════════════════════════════════════════════
def merge_with_existing(new_records: list[dict]) -> tuple[list[dict], int]:
    """
    รวม records ใหม่กับ jitarsa.json เดิม
    Strategy: ใช้ข้อมูลใหม่ทั้งหมด (replace) แทนที่จะ append
    เหตุผล: ข้อมูลงานเปลี่ยนแปลงได้ (วันที่ ที่นั่ง) จึง replace ดีกว่า append

    คืน (merged_records, count_added)
    count_added = จำนวน URL ใหม่ที่ไม่เคยมีในไฟล์เดิม (เพื่อ log)
    """
    existing = []
    if DATA_PATH.exists():
        with open(DATA_PATH, encoding="utf-8") as f:
            existing = json.load(f)

    # นับ URL ที่ไม่เคยมีในไฟล์เดิม (เพื่อแสดงใน summary)
    existing_urls = {r.get("url", "") for r in existing}
    added = [r for r in new_records if r.get("url", "") not in existing_urls]

    # replace ทั้งหมดด้วยข้อมูลใหม่ แล้ว re-index id
    merged = new_records
    for i, r in enumerate(merged):
        r["id"] = i

    return merged, len(added)


# ══════════════════════════════════════════════
# 6) SAVE + BACKUP — บันทึกไฟล์
# ══════════════════════════════════════════════
def save(records: list[dict]):
    """
    1. backup ไฟล์เก่าก่อน overwrite (เก็บ timestamp ไว้ใน filename)
    2. เขียน records ใหม่ลง jitarsa.json
    ensure_ascii=False: เก็บภาษาไทยเป็น UTF-8 ตรงๆ ไม่ escape เป็น \\uXXXX
    indent=2: อ่านได้ง่ายเวลา debug
    """
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    if DATA_PATH.exists():
        ts     = datetime.now().strftime("%Y%m%d_%H%M%S")  # เช่น 20260424_023000
        backup = BACKUP_DIR / f"jitarsa_{ts}.json"
        backup.write_text(DATA_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"  💾 backup → {backup.name}")

    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"  ✅ บันทึก {len(records)} รายการ → {DATA_PATH}")


# ══════════════════════════════════════════════
# 7) RELOAD VECTOR — แจ้ง main.py ให้ rebuild
# ══════════════════════════════════════════════
def reload_vector():
    """
    POST ไปที่ /reload-data ของ main.py (FastAPI)
    main.py จะโหลด jitarsa.json ใหม่และ rebuild FAISS vector
    timeout=120 เพราะการ rebuild อาจใช้เวลานาน (embedding ทุก doc)
    ถ้า server ยังไม่รัน → แจ้ง warning แต่ไม่ crash
    """
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
# MAIN — orchestrate ทั้งหมด
# ══════════════════════════════════════════════
async def main():
    print("=" * 50)
    print(f"🚀 JitArsa DB Update  —  {datetime.now():%Y-%m-%d %H:%M}")
    print("=" * 50)

    # ขั้นตอนที่ 1: scrape ข้อมูลใหม่จากเว็บ
    raw = await scrape_all()

    # ขั้นตอนที่ 2: ทำความสะอาดข้อมูล
    cleaned = clean_data(raw)

    # ขั้นตอนที่ 3: merge กับข้อมูลเดิม
    merged, added = merge_with_existing(cleaned)
    print(f"\n📊 สรุป: ทั้งหมด {len(merged)} รายการ (ใหม่ {added})")

    # ขั้นตอนที่ 4: บันทึกลงไฟล์
    print("\n💾 บันทึกไฟล์...")
    save(merged)

    # ขั้นตอนที่ 5: แจ้ง main.py ให้ rebuild vector
    print("\n🔄 แจ้ง server reload vector...")
    reload_vector()

    print("\n✅ อัพเดตเสร็จสมบูรณ์!")


if __name__ == "__main__":
    asyncio.run(main())