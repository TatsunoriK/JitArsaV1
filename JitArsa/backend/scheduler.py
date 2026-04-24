"""
scheduler.py
============
รัน update_database.py ทุกวัน เวลา 02:00
รัน: python scheduler.py

หรือใช้ Windows Task Scheduler / cron แทนก็ได้
"""

import schedule
import time
import subprocess
import sys
from pathlib import Path
from datetime import datetime

SCRIPT = Path(__file__).parent / "update_database.py"
LOG    = Path(__file__).parent / "data" / "update_log.txt"

def run_update():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{ts}] ▶ เริ่ม update database...")
    LOG.parent.mkdir(exist_ok=True)

    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*50}\n[{ts}] START\n")
        result = subprocess.run(
            [sys.executable, str(SCRIPT)],
            capture_output=True, text=True
        )
        f.write(result.stdout)
        if result.stderr:
            f.write("STDERR:\n" + result.stderr)
        status = "✅ SUCCESS" if result.returncode == 0 else f"❌ FAILED (code {result.returncode})"
        f.write(f"\n{status}\n")
        print(f"[{ts}] {status}")

# ─── ตั้งเวลา ───────────────────────────
schedule.every().day.at("02:00").do(run_update)   # รันทุกวัน 02:00
# schedule.every(12).hours.do(run_update)          # หรือ ทุก 12 ชม.
# schedule.every().monday.at("06:00").do(run_update) # หรือ ทุกวันจันทร์

print("⏰ Scheduler เริ่มทำงาน — รอรัน update_database.py ทุกวัน 02:00")
print("   กด Ctrl+C เพื่อหยุด\n")

# รันทันทีครั้งแรก (comment ออกถ้าไม่ต้องการ)
# run_update()

while True:
    schedule.run_pending()
    time.sleep(60)
