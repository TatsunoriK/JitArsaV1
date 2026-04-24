"""
scheduler.py
============
รัน update_database.py ตามกำหนดเวลาอัตโนมัติ
รัน: python scheduler.py  (แล้วปล่อยทิ้งไว้ background)

ทางเลือกอื่นแทน scheduler.py:
- Windows: Task Scheduler
- Linux/Mac: cron  (crontab -e แล้วเพิ่ม "0 2 * * * python /path/update_database.py")
"""

import schedule    # library จัดการ scheduled jobs (pip install schedule)
import time        # time.sleep(60) วน loop รอ
import subprocess  # รัน update_database.py เป็น subprocess
import sys         # ดึง path ของ Python interpreter ปัจจุบัน
from pathlib import Path
from datetime import datetime

# path ของ script ที่จะรัน (อยู่โฟลเดอร์เดียวกัน)
SCRIPT = Path(__file__).parent / "update_database.py"

# ไฟล์ log บันทึก output ของทุกครั้งที่รัน
LOG = Path(__file__).parent / "data" / "update_log.txt"


def run_update():
    """
    รัน update_database.py เป็น subprocess แล้วบันทึก output ลง log file

    ใช้ subprocess แทน import โดยตรงเพราะ:
    1. ได้ process ใหม่ที่แยกอิสระ — ถ้า crash ไม่กระทบ scheduler
    2. capture stdout/stderr ได้ครบ
    3. ตรวจ returncode ได้ว่าสำเร็จหรือไม่

    sys.executable = path ของ Python interpreter ที่รัน scheduler.py นี้อยู่
    ใช้ตัวเดียวกันกัน version conflict (เช่น python2 vs python3)
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{ts}] ▶ เริ่ม update database...")

    # สร้างโฟลเดอร์ data ถ้ายังไม่มี (กัน FileNotFoundError ตอนเขียน log)
    LOG.parent.mkdir(exist_ok=True)

    with open(LOG, "a", encoding="utf-8") as f:  # "a" = append ไม่ overwrite
        f.write(f"\n{'='*50}\n[{ts}] START\n")

        # รัน update_database.py และรอให้จบ
        # capture_output=True: เก็บ stdout และ stderr ไว้ใน result object
        # text=True: decode bytes เป็น string อัตโนมัติ
        result = subprocess.run(
            [sys.executable, str(SCRIPT)],
            capture_output=True,
            text=True
        )

        # บันทึก output ทั้งหมดลง log
        f.write(result.stdout)
        if result.stderr:
            f.write("STDERR:\n" + result.stderr)

        # returncode == 0 = สำเร็จ, อื่นๆ = error
        status = "✅ SUCCESS" if result.returncode == 0 else f"❌ FAILED (code {result.returncode})"
        f.write(f"\n{status}\n")
        print(f"[{ts}] {status}")


# ─── ตั้งเวลา ───────────────────────────────────────────
# schedule library: กำหนด job ที่จะรันตามเวลา
# .do(run_update) = ฟังก์ชันที่จะเรียกเมื่อถึงเวลา

schedule.every().day.at("02:00").do(run_update)    # ทุกวัน 02:00 น. (server load ต่ำ)
# schedule.every(12).hours.do(run_update)           # ทางเลือก: ทุก 12 ชั่วโมง
# schedule.every().monday.at("06:00").do(run_update) # ทางเลือก: ทุกวันจันทร์ 06:00

print("⏰ Scheduler เริ่มทำงาน — รอรัน update_database.py ทุกวัน 02:00")
print("   กด Ctrl+C เพื่อหยุด\n")

# รันทันทีครั้งแรกตอนเริ่ม (comment ออกถ้าไม่ต้องการ)
# run_update()

# ─── Main loop ───────────────────────────────────────────
# วน loop ตลอด ตรวจทุก 60 วินาทีว่ามี job ที่ถึงเวลาแล้วไหม
# schedule.run_pending() จะเรียก run_update() เมื่อเวลาตรง
# time.sleep(60) หยุด 60 วินาทีก่อนตรวจใหม่ (ลด CPU usage)
# ถ้า sleep 1 วินาที CPU จะทำงานหนักโดยไม่จำเป็น
while True:
    schedule.run_pending()
    time.sleep(60)