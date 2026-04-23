import httpx
import os
import requests
import pandas as pd
import json
import sys
import io
import logging
import httpx
import asyncio
import re
from datetime import datetime, date
from dotenv import load_dotenv

load_dotenv()

from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from transformers import logging as transformers_logging

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus.common import thai_stopwords

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
transformers_logging.set_verbosity_error()


def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        print(text.encode('utf-8', errors='replace').decode('ascii', errors='replace'))


# ===============================
# 1) FASTAPI
# ===============================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# 2) CONFIG
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data/jitarsa.json")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_TIMEOUT = 60

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# ===============================
# 3) NLP SETUP
# ===============================
stopwords = set(thai_stopwords()) - {
    "กิจกรรม", "โครงการ", "สมัคร", "อาสาสมัคร"
}

PROVINCE_ALIAS = {
    "กทม": "กรุงเทพ",
    "กรุงเทพฯ": "กรุงเทพ",
    "กรุงเทพมหานคร": "กรุงเทพ",
    "bangkok": "กรุงเทพ",
    "bkk": "กรุงเทพ",
    "โคราช": "นครราชสีมา",
    "อยุธยา": "พระนครศรีอยุธยา",
}

ALL_PROVINCES = [
    "กรุงเทพ",
    "กระบี่", "กาญจนบุรี", "กาฬสินธุ์", "กำแพงเพชร",
    "ขอนแก่น", "จันทบุรี", "ฉะเชิงเทรา", "ชลบุรี",
    "ชัยนาท", "ชัยภูมิ", "ชุมพร", "เชียงราย", "เชียงใหม่",
    "ตรัง", "ตราด", "ตาก", "นครนายก", "นครปฐม",
    "นครพนม", "นครราชสีมา", "นครศรีธรรมราช",
    "นครสวรรค์", "นนทบุรี", "นราธิวาส", "น่าน",
    "บึงกาฬ", "บุรีรัมย์", "ปทุมธานี", "ประจวบคีรีขันธ์",
    "ปราจีนบุรี", "ปัตตานี", "พระนครศรีอยุธยา",
    "พะเยา", "พังงา", "พัทลุง", "พิจิตร", "พิษณุโลก",
    "เพชรบุรี", "เพชรบูรณ์", "แพร่", "ภูเก็ต",
    "มหาสารคาม", "มุกดาหาร", "แม่ฮ่องสอน",
    "ยโสธร", "ยะลา", "ร้อยเอ็ด", "ระนอง", "ระยอง",
    "ราชบุรี", "ลพบุรี", "ลำปาง", "ลำพูน", "เลย",
    "ศรีสะเกษ", "สกลนคร", "สงขลา", "สตูล", "สมุทรปราการ",
    "สมุทรสงคราม", "สมุทรสาคร", "สระแก้ว", "สระบุรี",
    "สิงห์บุรี", "สุโขทัย", "สุพรรณบุรี", "สุราษฎร์ธานี",
    "สุรินทร์", "หนองคาย", "หนองบัวลำภู", "อ่างทอง",
    "อำนาจเจริญ", "อุดรธานี", "อุตรดิตถ์", "อุทัยธานี",
    "อุบลราชธานี",
]

docs = None
retriever = None

# ===============================
# 4) REQUEST MODEL
# ===============================

class HistoryMessage(BaseModel):
    role: str
    content: str


class QuestionRequest(BaseModel):
    question: str
    history: Optional[List[HistoryMessage]] = []

# ===============================
# 5) INTENT DETECTION
# ===============================

# คีย์เวิร์ดที่บ่งบอกว่าต้องการค้นหางาน
SEARCH_KEYWORDS = [
    "มีงาน", "หางาน", "แนะนำงาน", "งานอาสา", "กิจกรรม", "โครงการ",
    "สมัคร", "ฟรี", "ออนไลน์", "ทำที่บ้าน", "เสาร์", "อาทิตย์",
    "แถว", "ใกล้", "ที่ไหน", "แนวไหน", "ช่วยเด็ก", "ช่วยผู้สูงอายุ",
    "สิ่งแวดล้อม", "สัตว์", "สอน", "ปลูก", "ทำความสะอาด",
]

# คีย์เวิร์ดที่บ่งบอกว่าเป็นคำถามทั่วไป ไม่ต้องดึง context งาน
GENERAL_KEYWORDS = [
    "สวัสดี", "หวัดดี", "ดีจ้า", "hello", "hi",
    "ชื่ออะไร", "คุณคือ", "เป็นใคร", "ทำอะไร",
    "จิตอาสาคืออะไร", "อาสาสมัครคืออะไร", "ทำไมต้อง",
    "กลัว", "เหงา", "ไม่เคย", "ขอบคุณ", "บ๊ายบาย", "ลาก่อน",
]

def detect_intent(question: str, history: list) -> str:
    """
    คืนค่า: 'search' | 'general' | 'clarify'
    - search: ต้องการค้นหางาน → ดึง context
    - general: คำถามทั่วไป → ไม่ต้องดึง context
    - clarify: ยังไม่ชัดเจน → ให้ LLM ถามกลับ
    """
    q = question.lower().strip()

    # ตรวจ province ก่อน → ถ้าระบุจังหวัดมา = search แน่ๆ
    if detect_province_in_query(question):
        return "search"

    # ตรวจ general keywords
    for kw in GENERAL_KEYWORDS:
        if kw in q:
            return "general"

    # ตรวจ search keywords
    for kw in SEARCH_KEYWORDS:
        if kw in q:
            return "search"

    # ดู history ว่ามีการถามเรื่องงานก่อนหน้าไหม → ถ้ามีแล้วมาตอบต่อ = search
    if history:
        recent = " ".join([h.content for h in history[-4:] if h.role == "user"]).lower()
        for kw in SEARCH_KEYWORDS + ALL_PROVINCES:
            if kw in recent:
                return "search"

    # default: ให้ LLM จัดการเองโดยไม่ดึง context
    return "general"


# ===============================
# 6) HELPERS
# ===============================

def extract_provinces(text: str) -> list:
    if not text or pd.isna(text):
        return []
    text = normalize_text(str(text))
    found = []
    for prov in ALL_PROVINCES:
        if prov in str(text):
            canonical = PROVINCE_ALIAS.get(prov, prov)
            if canonical not in found:
                found.append(canonical)
    return found


def clean(text):
    if pd.isna(text) or text is None:
        return ""
    text = str(text).strip().lower()
    for k, v in PROVINCE_ALIAS.items():
        text = text.replace(k.lower(), v.lower())
    tokens = word_tokenize(text, engine="newmm")
    tokens = [t for t in tokens if t.strip() and t not in stopwords]
    return " ".join(tokens)


def normalize_text(text):
    if pd.isna(text) or text is None:
        return ""
    text = str(text).strip()
    for k, v in PROVINCE_ALIAS.items():
        text = text.replace(k, v)
    return text


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"ไม่พบไฟล์ {path}")
    return pd.read_json(path) if path.endswith(".json") else pd.read_csv(path)


# ===============================
# วันที่: parse + กรองหมดอายุ
# ===============================
THAI_MONTHS = {
    "มกราคม": 1, "กุมภาพันธ์": 2, "มีนาคม": 3, "เมษายน": 4,
    "พฤษภาคม": 5, "มิถุนายน": 6, "กรกฎาคม": 7, "สิงหาคม": 8,
    "กันยายน": 9, "ตุลาคม": 10, "พฤศจิกายน": 11, "ธันวาคม": 12,
    "ม.ค.": 1, "ก.พ.": 2, "มี.ค.": 3, "เม.ย.": 4,
    "พ.ค.": 5, "มิ.ย.": 6, "ก.ค.": 7, "ส.ค.": 8,
    "ก.ย.": 9, "ต.ค.": 10, "พ.ย.": 11, "ธ.ค.": 12,
}

def parse_event_end_date(date_str: str) -> date | None:
    """
    พยายาม parse วันที่สิ้นสุดของงานจาก string เช่น
    "10:00 เสาร์ 4 เม.ย. 2569 - 18:00 เสาร์ 4 เม.ย. 2569"
    "วันที่ 27 เม.ย. 2569 และ 24 เม.ย. 2569"
    คืน date object ของวันสุดท้าย หรือ None ถ้า parse ไม่ได้
    """
    if not date_str or date_str == "ไม่ระบุ":
        return None
    try:
        # หาตัวเลขวัน + เดือน + ปีพุทธศักราช ทุกชุดในสตริง
        pattern = r"(\d{1,2})\s+(" + "|".join(re.escape(m) for m in THAI_MONTHS) + r")\s+(\d{4})"
        matches = re.findall(pattern, date_str)
        if not matches:
            return None
        dates = []
        for day_s, month_s, year_s in matches:
            month = THAI_MONTHS.get(month_s)
            if not month:
                continue
            day = int(day_s)
            year_be = int(year_s)
            year_ce = year_be - 543  # แปลง พ.ศ. → ค.ศ.
            dates.append(date(year_ce, month, day))
        return max(dates) if dates else None
    except Exception:
        return None


def is_expired(date_str: str, today: date | None = None) -> bool:
    """คืน True ถ้างานหมดแล้ว (วันสุดท้ายผ่านมาแล้ว)"""
    if today is None:
        today = datetime.now().date()
    end = parse_event_end_date(date_str)
    if end is None:
        return False  # parse ไม่ได้ → เก็บไว้ก่อน ไม่กรองทิ้ง
    return end < today


def preprocess(df):
    df = df.fillna("ไม่ระบุ").copy()

    col_map = {
        "ชื่อกิจกรรม": "title",
        "ชื่อองค์กร": "org",
        "สถานที่": "location",
        "วันที่-เวลา": "date",
        "url": "url",
    }
    for json_col, alias in col_map.items():
        df[alias] = df[json_col].apply(
            normalize_text) if json_col in df.columns else "ไม่ระบุ"

    if "มีค่าใช้จ่าย" in df.columns:
        df["cost"] = df["มีค่าใช้จ่าย"].apply(
            lambda x: "ไม่เสียค่าใช้จ่าย"
            if x is False or str(x).lower() in ["false", "0", "ไม่ระบุ"]
            else "มีค่าใช้จ่าย"
        )
    else:
        df["cost"] = "ไม่ระบุ"

    def get_provinces(row):
        combined = " ".join([
            str(row.get("ชื่อกิจกรรม", "")),
            str(row.get("สถานที่", "")),
            str(row.get("รายละเอียด", ""))[:1000],
        ])
        return " ".join(extract_provinces(combined))

    df["provinces"] = df.apply(get_provinces, axis=1)

    def fix_location(row):
        if row["location"] == "ไม่ระบุ" and row["provinces"]:
            return row["provinces"]
        return row["location"]

    df["location"] = df.apply(fix_location, axis=1)

    for c in ["title", "org", "location", "date", "cost"]:
        df[f"{c}_clean"] = df[c].apply(clean)

    df["provinces_clean"] = df["provinces"].apply(clean)

    def combine_search(r):
        return " ".join([
            str(r.get("title_clean", "")),
            str(r.get("org_clean", "")),
            str(r.get("location_clean", "")),
            str(r.get("date_clean", "")),
            str(r.get("cost_clean", "")),
            str(r.get("provinces_clean", "")),
        ]).strip()

    df["search_text"] = df.apply(combine_search, axis=1)
    return df


def build_vector(df):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )

    all_docs = []

    for _, row in df.iterrows():
        metadata = {
            "title": row.get("title", "ไม่ระบุ"),
            "org": row.get("org", "ไม่ระบุ"),
            "location": row.get("location", "ไม่ระบุ"),
            "date": row.get("date", "ไม่ระบุ"),
            "cost": row.get("cost", "ไม่ระบุ"),
            "provinces": row.get("provinces", ""),
            "url": row.get("url", "ไม่ระบุ"),
            "doc_type": "main",
        }

        all_docs.append(Document(
            page_content=row["search_text"],
            metadata=metadata,
        ))

        detail = str(row.get("รายละเอียด", "")).strip()
        if detail and detail != "ไม่ระบุ":
            chunks = splitter.split_text(detail)
            for i, chunk in enumerate(chunks):
                chunk_meta = {**metadata, "doc_type": "detail", "chunk_index": i}
                all_docs.append(Document(
                    page_content=chunk,
                    metadata=chunk_meta,
                ))

    print(f"total docs (main + chunks) = {len(all_docs)}")

    emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.from_documents(all_docs, emb)
    local_retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 20, "fetch_k": 80, "lambda_mult": 0.6}
    )
    return all_docs, local_retriever


def detect_province_in_query(q: str):
    q_norm = normalize_text(q)
    for prov in ALL_PROVINCES:
        if prov in q_norm:
            return PROVINCE_ALIAS.get(prov, prov)
    return None


def enhance_query(q):
    q = normalize_text(q).strip()
    extra = []
    if "ฟรี" in q:
        extra += ["ไม่เสียค่าใช้จ่าย", "ไม่มีค่าใช้จ่าย"]
    if "ออนไลน์" in q:
        extra += ["online", "remote", "ทำที่บ้าน", "work from home"]
    if "กรุงเทพ" in q:
        extra += ["กรุงเทพมหานคร", "กทม", "bangkok"]
    if extra:
        q = q + " " + " ".join(extra)
    tokens = word_tokenize(q, engine="newmm")
    tokens = [t for t in tokens if t.strip() and t not in stopwords]
    return " ".join(tokens)


def filter_docs(found_docs, q, locked_province: str = None):
    q_norm = normalize_text(q).lower()
    want_free = "ฟรี" in q_norm
    want_online = "ออนไลน์" in q_norm or "ทำที่บ้าน" in q_norm
    want_not_online = "ไม่ออนไลน์" in q_norm or "ออนไซต์" in q_norm
    want_province = locked_province or detect_province_in_query(q)
    online_keywords = ["ออนไลน์", "online", "remote", "ทำที่บ้าน", "work from home"]
    today = datetime.now().date()

    if want_province:
        global docs
        results = []
        for d in docs:
            if d.metadata.get("doc_type") != "main":
                continue
            md = d.metadata
            # กรองงานหมดอายุ
            if is_expired(md.get("date", ""), today):
                continue
            merged = " ".join([
                str(md.get("provinces", "")),
                str(md.get("location", "")),
                str(md.get("title", "")),
                (d.page_content or ""),
            ]).lower()
            if want_province.lower() not in merged:
                continue
            is_free = "ไม่เสียค่าใช้จ่าย" in str(md.get("cost", "")).lower()
            is_online = any(k in merged for k in online_keywords)
            if want_free and not is_free:
                continue
            if want_online and not is_online:
                continue
            if want_not_online and is_online:
                continue
            results.append(d)
        print(f"พบงานใน {want_province} (ไม่หมดอายุ) = {len(results)} งาน")
        return results

    result = []
    for d in found_docs:
        md = d.metadata
        # กรองงานหมดอายุ
        if is_expired(md.get("date", ""), today):
            continue
        merged = " ".join([
            (d.page_content or ""),
            str(md.get("title", "")),
            str(md.get("location", "")),
            str(md.get("cost", "")),
        ]).lower()
        is_free = "ไม่เสียค่าใช้จ่าย" in merged
        is_online = any(k in merged for k in online_keywords)
        if want_free and not is_free:
            continue
        if want_online and not is_online:
            continue
        if want_not_online and is_online:
            continue
        result.append(d)

    has_hard = want_free or want_online or want_not_online
    if len(result) < 2 and not has_hard:
        return [d for d in found_docs if not is_expired(d.metadata.get("date", ""), today)]
    return result


def deduplicate_docs(found_docs):
    seen = set()
    unique = []
    for d in found_docs:
        md = d.metadata
        key = md.get("url", "") or md.get("title", "")
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


def build_context(found_docs, max_items=5) -> str:
    lines = []
    count = 0
    for d in found_docs:
        if count >= max_items:
            break
        md = d.metadata
        title = md.get("title", "")
        if not title or title == "ไม่ระบุ":
            continue
        count += 1
        lines.append(
            f"[งานที่ {count}]\n"
            f"ชื่อ: {title}\n"
            f"องค์กร: {md.get('org', 'ไม่ระบุ')}\n"
            f"สถานที่: {md.get('location', 'ไม่ระบุ')}\n"
            f"วันที่: {md.get('date', 'ไม่ระบุ')}\n"
            f"ค่าใช้จ่าย: {md.get('cost', 'ไม่ระบุ')}\n"
            f"ลิงก์: {md.get('url', 'ไม่ระบุ')}\n"
        )
    return "\n".join(lines)


# ===============================
# SYSTEM PROMPT (ปรับให้ strict ขึ้น)
# ===============================
SYSTEM_PERSONA = """
คุณชื่อ "ภา" (นพนภา) เป็นผู้ช่วยหางานอาสาสมัครเท่านั้น

โทน: เป็นกันเอง สุภาพ เหมือนเพื่อน ใช้คำลงท้าย "ค่ะ" แทนตัวเองว่า "ภา"

=== กฎเหล็ก (ห้ามฝ่าฝืนเด็ดขาด) ===
1. ตอบสั้นเสมอ (1-3 ประโยค ยกเว้นแสดงรายการงาน)
2. ห้ามแต่งข้อมูลงานขึ้นมาเอง — ใช้เฉพาะข้อมูลจาก [งานอาสาที่เกี่ยวข้อง] เท่านั้น
3. ถ้า [งานอาสาที่เกี่ยวข้อง] ว่างเปล่า → ห้ามโชว์งาน บอกว่าหาไม่เจอแล้วเสนอถามเพิ่ม
4. ห้ามพูดถึง AI / ระบบ / โมเดล / algorithm
5. ถ้าคำถามไม่เกี่ยวกับงานอาสา → ปฏิเสธสุภาพ 1 ประโยค แล้วชวนกลับเรื่องงาน
6. จำบริบทจาก history เสมอ — ถ้าผู้ใช้บอกพื้นที่/แนวงานไปแล้ว ห้ามถามซ้ำ

=== การค้นหางาน ===
- คำถามกว้าง (ไม่ระบุแนวหรือพื้นที่) → ถามกลับ 1 คำถามก่อน ห้ามแสดงงานทันที
- คำถามชัดเจน (ระบุแนว + พื้นที่ หรือระบุอย่างใดอย่างหนึ่งชัดๆ) → แสดงงานได้เลย ไม่เกิน 5 งาน

รูปแบบแสดงงาน (ใช้แบบนี้เท่านั้น ห้ามดัดแปลง):
- [ชื่องาน]
  📅 [วันที่] | 📍 [สถานที่]
  🔗 [ลิงก์]

=== ตัวอย่าง ===

Q: หวัดดี
A: หวัดดีค่ะ 😊 อยากให้ภาช่วยหางานอาสาไหมคะ?

Q: มีงานอาสาไหม
A: มีค่ะ สนใจแนวไหน หรืออยากทำแถวไหนบอกภาได้เลยนะคะ 😊

Q: มีงานแถวเชียงใหม่ช่วยเด็กไหม
A: [แสดงงานจาก context ที่ได้รับเท่านั้น]

Q: อยากกินข้าว
A: ฮ่าๆ ภาช่วยได้แค่เรื่องงานอาสานะคะ 😅 มีอะไรอยากลองทำไหมคะ?

Q: ทำข้อความไม่ต่อเนื่อง
A: ภาไม่แน่ใจว่าหมายถึงอะไรเลยค่ะ อยากให้ภาช่วยหางานอาสาไหมคะ?
"""


def build_groq_messages(question: str, context: str, history: list) -> list:
    messages = [{"role": "system", "content": SYSTEM_PERSONA}]

    # ใส่ history ทั้งหมด (ไม่ตัดทิ้ง) เพื่อให้ LLM จำบริบท
    for h in (history or []):
        role = "assistant" if h.role != "user" else "user"
        messages.append({"role": role, "content": h.content})

    # คำถามปัจจุบัน + context (ถ้ามี)
    if context:
        user_msg = (
            f"คำถาม: {question}\n\n"
            f"[งานอาสาที่เกี่ยวข้อง]\n{context}\n\n"
            f"ตอบโดยใช้ข้อมูลจาก [งานอาสาที่เกี่ยวข้อง] เท่านั้น ห้ามแต่งข้อมูลเพิ่ม"
        )
    else:
        user_msg = question

    messages.append({"role": "user", "content": user_msg})
    return messages


async def groq_stream_generator(question: str, context: str, history: list):
    messages = build_groq_messages(question, context, history)
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": GROQ_MODEL,
        "messages": messages,
        "max_tokens": 600,
        "temperature": 0.5,   # ลด temperature → ตอบมั่วน้อยลง
        "stream": True,
    }
    async with httpx.AsyncClient(timeout=GROQ_TIMEOUT) as client:
        async with client.stream(
            "POST", "https://api.groq.com/openai/v1/chat/completions",
            headers=headers, json=body
        ) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if line.startswith("data:"):
                    payload = line[5:].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(payload)
                        text = chunk["choices"][0]["delta"].get("content", "")
                        if text:
                            yield text
                    except Exception:
                        continue


def extract_province_from_history(history: list) -> str | None:
    """
    ดึงจังหวัดล่าสุดที่ผู้ใช้เคยระบุใน history
    เพื่อ lock จังหวัดข้ามเทิร์น
    """
    for h in reversed(history or []):
        if h.role != "user":
            continue
        prov = detect_province_in_query(h.content)
        if prov:
            return prov
    return None


def ask_rag(question: str, history: list = None) -> tuple[str, list]:
    """
    Returns (context_str, found_docs)
    - ใช้ intent detection ก่อนเสมอ
    - lock จังหวัดจาก history ถ้าคำถามปัจจุบันไม่ได้ระบุจังหวัดใหม่
    - กรองงานหมดอายุออกทุกกรณี
    """
    global docs, retriever

    intent = detect_intent(question, history or [])
    print(f"intent = {intent}")

    if intent == "general":
        return "", []

    # ตรวจจังหวัดจากคำถามปัจจุบันก่อน ถ้าไม่มีให้ดูจาก history
    current_province = detect_province_in_query(question)
    locked_province = current_province or extract_province_from_history(history or [])
    print(f"province: current={current_province}, locked={locked_province}")

    if locked_province:
        print(f"Province mode: {locked_province}")
        found = filter_docs([], question, locked_province=locked_province)
    else:
        # รวม history ล่าสุดเข้า query เพื่อให้จำบริบท
        context_query = question
        if history:
            prev_user = " ".join([
                h.content for h in history[-6:]
                if h.role == "user"
            ])
            context_query = prev_user + " " + question

        query = enhance_query(context_query)
        found = retriever.invoke(query)
        print(f"question={question}, enhanced={query}, before filter={len(found)}")
        found = filter_docs(found, context_query)

    found = deduplicate_docs(found)
    print(f"after dedup = {len(found)}")
    context = build_context(found)
    return context, found


# ===============================
# 7) STARTUP
# ===============================
@app.on_event("startup")
def startup_event():
    global docs, retriever
    df = preprocess(load_data(DATASET_PATH))
    docs, retriever = build_vector(df)
    print("โหลดข้อมูลและสร้าง vector database เรียบร้อยแล้ว")


# ===============================
# 8) ROUTES
# ===============================
@app.get("/")
def root():
    return {"message": "JitArsa backend is running"}


@app.post("/ask-pha")
async def ask_api(data: QuestionRequest):
    try:
        history = data.history or []

        # DEBUG: ตรวจว่า history ที่รับมาครบไหม
        print(f"[DEBUG] question={data.question!r}")
        print(f"[DEBUG] history len={len(history)}")
        for i, h in enumerate(history):
            print(f"[DEBUG]   history[{i}] role={h.role!r} content={h.content[:60]!r}")

        context, _ = ask_rag(data.question, history)

        async def stream_response():
            async for chunk in groq_stream_generator(
                data.question, context, history
            ):
                yield chunk

        return StreamingResponse(
            stream_response(),
            media_type="text/plain; charset=utf-8"
        )

    except httpx.TimeoutException:
        return {"answer": "ภาคิดช้าไปหน่อย ลองถามใหม่นะคะ 🙏"}
    except httpx.HTTPStatusError as e:
        print(f"HTTP error: {e.response.status_code} - {e.response.text}")
        return {"answer": "เชื่อมต่อไม่ได้ ลองใหม่อีกทีนะคะ"}
    except Exception as e:
        print(f"Error: {e}")
        return {"answer": f"เกิดข้อผิดพลาด: {str(e)}"}