import httpx

keys = {
    "KEY1": "AIzaSyCMtcatcvp72MCrKllIKtBXB1gb1iX49Gg",
    "KEY2": "AIzaSyAQFNRb2lCzssE0UP7TQ575OUMSEayo5yw",
    "KEY3": "AIzaSyB0r8FTitraX-2-cv1JVZNTlqCh0JDpy4U",
}

for name, key in keys.items():
    r = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent?key={key}",
        json={"contents": [{"parts": [{"text": "สวัสดี"}]}]}
    )
    status = r.status_code
    if status == 200:
        print(f"{name}: ✅ ใช้ได้")
    elif status == 429:
        print(f"{name}: ❌ quota หมด")
    else:
        print(f"{name}: ⚠️ status {status}")
