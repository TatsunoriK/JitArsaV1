import httpx
import os
import pandas as pd
import json
import sys
import io
import httpx
import re
from datetime import datetime, date
from dotenv import load_dotenv
import itertools

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

# ✅ Key rotation — วนไปเรื่อยๆ เมื่อ key ไหนหมด/rate limit สลับไปตัวถัดไปอัตโนมัติ
_RAW_KEYS = os.environ.get("GROQ_API_KEYS", os.environ.get("GROQ_API_KEY", ""))
GROQ_API_KEYS = [k.strip() for k in _RAW_KEYS.split(",") if k.strip()]
if not GROQ_API_KEYS:
    raise RuntimeError("ไม่พบ GROQ_API_KEY ใน .env")
_key_cycle = itertools.cycle(GROQ_API_KEYS)
_current_key_index = 0

def get_next_groq_key() -> str:
    """เลือก key ถัดไปในวงจร"""
    global _current_key_index
    _current_key_index = (_current_key_index + 1) % len(GROQ_API_KEYS)
    key = GROQ_API_KEYS[_current_key_index]
    print(f"[GROQ] สลับไปใช้ key #{_current_key_index + 1} (****{key[-6:]})")
    return key

def get_current_groq_key() -> str:
    return GROQ_API_KEYS[_current_key_index]

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

# ✅ mapping ภาค/โซน → รายชื่อจังหวัด
REGION_MAP = {
    "เหนือ":    ["เชียงใหม่","เชียงราย","ลำปาง","ลำพูน","แม่ฮ่องสอน","พะเยา","แพร่","น่าน","อุตรดิตถ์","ตาก","สุโขทัย","พิษณุโลก","พิจิตร","กำแพงเพชร","เพชรบูรณ์"],
    "ใต้":      ["สงขลา","สุราษฎร์ธานี","นครศรีธรรมราช","ภูเก็ต","กระบี่","พังงา","ตรัง","พัทลุง","สตูล","ระนอง","ปัตตานี","ยะลา","นราธิวาส","ชุมพร"],
    "กลาง":     ["กรุงเทพ","นนทบุรี","ปทุมธานี","สมุทรปราการ","สมุทรสาคร","สมุทรสงคราม","นครปฐม","สุพรรณบุรี","กาญจนบุรี","ราชบุรี","เพชรบุรี","ประจวบคีรีขันธ์","อยุธยา","อ่างทอง","สิงห์บุรี","ชัยนาท","ลพบุรี","สระบุรี","นครนายก","ปราจีนบุรี"],
    "ออก":      ["ชลบุรี","ระยอง","จันทบุรี","ตราด","ฉะเชิงเทรา","สระแก้ว"],
    "ตะวันออก": ["ชลบุรี","ระยอง","จันทบุรี","ตราด","ฉะเชิงเทรา","สระแก้ว"],
    "อีสาน":    ["นครราชสีมา","ขอนแก่น","อุดรธานี","อุบลราชธานี","บุรีรัมย์","สุรินทร์","ศรีสะเกษ","มหาสารคาม","ร้อยเอ็ด","กาฬสินธุ์","สกลนคร","นครพนม","มุกดาหาร","อำนาจเจริญ","ยโสธร","ชัยภูมิ","เลย","หนองคาย","หนองบัวลำภู","บึงกาฬ","อุทัยธานี"],
    "อีสานเหนือ": ["อุดรธานี","หนองคาย","บึงกาฬ","นครพนม","สกลนคร","มุกดาหาร","หนองบัวลำภู","เลย"],
    "ตะวันตก":  ["กาญจนบุรี","ราชบุรี","เพชรบุรี","ประจวบคีรีขันธ์","ตาก"],
}
# alias สำหรับคำพูดปกติ → ชื่อ key ใน REGION_MAP
REGION_ALIAS = {
    "ภาคเหนือ": "เหนือ", "โซนเหนือ": "เหนือ", "แถบเหนือ": "เหนือ", "ทางเหนือ": "เหนือ",
    "ภาคใต้": "ใต้", "โซนใต้": "ใต้", "แถบใต้": "ใต้", "ทางใต้": "ใต้",
    "ภาคกลาง": "กลาง", "โซนกลาง": "กลาง", "แถบกลาง": "กลาง",
    "ภาคตะวันออก": "ออก", "ภาคออก": "ออก", "โซนออก": "ออก", "อีสเทิร์น": "ออก", "อีสเทอร์น": "ออก",
    "ภาคอีสาน": "อีสาน", "โซนอีสาน": "อีสาน", "ภาคตะวันออกเฉียงเหนือ": "อีสาน", "ทางอีสาน": "อีสาน",
    "ภาคตะวันตก": "ตะวันตก", "โซนตะวันตก": "ตะวันตก",
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
    # ค้นหาทั่วไป
    "มีงาน", "หางาน", "แนะนำงาน", "งานอาสา", "จิตอาสา", "กิจกรรม", "โครงการ",
    "สมัคร", "ฟรี", "ออนไลน์", "ทำที่บ้าน", "เสาร์", "อาทิตย์",
    "แถว", "ใกล้", "ที่ไหน", "แนวไหน", "อยาก", "ถนัด", "ชอบ", "สนใจ",
    # กลุ่มเป้าหมาย
    "ช่วยเด็ก", "ช่วยผู้สูงอายุ", "ผู้พิการ", "คนไร้บ้าน", "ผู้ป่วย",
    "สัตว์", "ชุมชน", "ผู้ด้อยโอกาส", "เยาวชน", "นักเรียน",
    # ประเภทงาน / ทักษะ
    "สอน", "ติวเตอร์", "ครู", "การศึกษา",
    "ก่อสร้าง", "ซ่อม", "ช่าง", "งานหนัก", "แรงงาน",
    "ถักผ้า", "ถักนิตติ้ง", "เย็บ", "งานฝีมือ", "หัตถกรรม", "งานประดิษฐ์",
    "บริหาร", "จัดการ", "ประสานงาน", "เอกสาร", "ออฟฟิศ", "สำนักงาน",
    "ออกแบบ", "กราฟิก", "ศิลปะ", "วาดรูป", "ถ่ายภาพ", "วิดีโอ", "ตัดต่อ",
    "ดนตรี", "ร้องเพลง", "กีฬา", "นาฏศิลป์", "การแสดง",
    "แพทย์", "พยาบาล", "สาธารณสุข", "ปฐมพยาบาล", "สุขภาพ",
    "IT", "โปรแกรม", "คอมพิวเตอร์", "เทคโนโลยี", "โค้ด",
    "แปล", "ล่าม", "ภาษา", "อังกฤษ", "จีน", "ญี่ปุ่น",
    "ทำอาหาร", "ทำครัว", "เบเกอรี่",
    "ปลูก", "เกษตร", "ต้นไม้", "ป่า", "ทะเล",
    "ทำความสะอาด", "เก็บขยะ", "สิ่งแวดล้อม",
    "ขับรถ", "ส่งของ", "โลจิสติกส์",
    "โซเชียล", "PR", "ประชาสัมพันธ์", "การตลาด",
    "ระยะสั้น", "ระยะยาว", "วันเดียว", "ค่าย", "อยู่ค่าย",
]

# คีย์เวิร์ดที่บ่งบอกว่าเป็นคำถามทั่วไป ไม่ต้องดึง context งาน
GENERAL_KEYWORDS = [
    # ทักทาย
    "สวัสดี", "หวัดดี", "ดีจ้า", "hello", "hi", "ขอบคุณ", "บ๊ายบาย", "ลาก่อน",
    # ถามเกี่ยวกับ AI / ตัวตน
    "ชื่ออะไร", "คุณคือ", "เป็นใคร", "ทำอะไร",
    # คำถามทั่วไปเกี่ยวกับจิตอาสา (ไม่ต้องค้นงาน)
    "จิตอาสาคืออะไร", "อาสาสมัครคืออะไร", "ทำไมต้องทำ", "ประโยชน์",
    "เริ่มยังไง", "เริ่มต้นยังไง", "มือใหม่", "ครั้งแรก",
    "เตรียมตัวยังไง", "ต้องเตรียม", "ควรรู้อะไร",
    "ต่างกันยังไง", "อาสากับ", "หมายความว่า",
    "กลัว", "เหงา", "ไม่เคย", "ประสบการณ์", "ได้อะไร",
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
    # ✅ ตรวจภาค/โซน
    if detect_region_in_query(question):
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
        "รายละเอียด": "detail",  # ✅ เพิ่ม detail
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
        # ✅ รวม รายละเอียด เข้าด้วย เพื่อให้ค้นหาตามทักษะ/ประเภทงานได้
        detail_text = clean(str(r.get("detail", "")))[:300]  # ตัดให้สั้น
        return " ".join([
            str(r.get("title_clean", "")),
            str(r.get("org_clean", "")),
            str(r.get("location_clean", "")),
            str(r.get("date_clean", "")),
            str(r.get("cost_clean", "")),
            str(r.get("provinces_clean", "")),
            detail_text,
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


def detect_region_in_query(q: str):
    """คืนค่า list of provinces ถ้าพบคำบอกภาค/โซน"""
    q_lower = q.lower().strip()
    # ตรวจ alias ก่อน (เช่น "ภาคเหนือ", "โซนใต้")
    for alias, region_key in REGION_ALIAS.items():
        if alias in q_lower:
            return REGION_MAP.get(region_key, [])
    # ตรวจชื่อ region โดยตรง (เช่น "เหนือ", "ใต้")
    for region_key, provinces in REGION_MAP.items():
        if region_key in q_lower:
            return provinces
    return []

def detect_province_in_query(q: str):
    q_norm = normalize_text(q)
    for prov in ALL_PROVINCES:
        if prov in q_norm:
            return PROVINCE_ALIAS.get(prov, prov)
    # ✅ ตรวจภาค/โซน → คืนจังหวัดแรกในภาคนั้น (ใช้ filter_docs_by_region แทน)
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


# ✅ mapping ทักษะ/ความสนใจ → keywords ที่จะใช้ filter งาน
SKILL_KEYWORDS = {
    "ก่อสร้าง":    ["ก่อสร้าง", "สร้าง", "ซ่อม", "ซ่อมแซม", "ช่าง", "ทาสี", "ปรับปรุง", "อาคาร"],
    "ช่าง":        ["ช่าง", "ซ่อม", "ซ่อมแซม", "ก่อสร้าง", "ทาสี", "ติดตั้ง"],
    "ถักผ้า":      ["ถัก", "ผ้า", "เย็บ", "ถักนิตติ้ง", "งานฝีมือ", "หัตถกรรม", "ประดิษฐ์"],
    "ถักนิตติ้ง":  ["ถัก", "นิตติ้ง", "ผ้า", "เย็บ", "งานฝีมือ"],
    "เย็บ":        ["เย็บ", "ผ้า", "ถัก", "งานฝีมือ"],
    "งานฝีมือ":    ["ฝีมือ", "หัตถกรรม", "ประดิษฐ์", "ถัก", "เย็บ", "ทำมือ"],
    "บริหาร":      ["จัดการ", "ประสานงาน", "อีเวนต์", "กิจกรรม", "จัดงาน", "โครงการ"],
    "ประสานงาน":   ["ประสานงาน", "จัดการ", "อีเวนต์", "จัดงาน"],
    "สอน":         ["สอน", "ติวเตอร์", "การศึกษา", "ครู", "เด็ก", "ค่าย"],
    "ออกแบบ":      ["ออกแบบ", "กราฟิก", "ศิลปะ", "วาดรูป", "สื่อ"],
    "ถ่ายภาพ":     ["ถ่ายภาพ", "ภาพ", "วิดีโอ", "สื่อ", "ตัดต่อ"],
    "ดนตรี":       ["ดนตรี", "ร้องเพลง", "เพลง", "ดนตรีบำบัด"],
    "กีฬา":        ["กีฬา", "ฟุตบอล", "วิ่ง", "กีฬาบำบัด", "เยาวชน"],
    "ศิลปะ":       ["ศิลปะ", "วาดรูป", "เพ้นท์", "ระบายสี", "ออกแบบ"],
    "แพทย์":       ["แพทย์", "พยาบาล", "สุขภาพ", "ปฐมพยาบาล", "สาธารณสุข"],
    "พยาบาล":      ["พยาบาล", "แพทย์", "สุขภาพ", "ดูแลผู้ป่วย"],
    "ทำอาหาร":     ["อาหาร", "ครัว", "เลี้ยงอาหาร", "ทำครัว"],
    "เกษตร":       ["เกษตร", "ปลูก", "ต้นไม้", "ป่า", "สวน"],
    "สิ่งแวดล้อม": ["สิ่งแวดล้อม", "ปลูกป่า", "ทำความสะอาด", "เก็บขยะ", "ทะเล"],
    "IT":          ["IT", "คอมพิวเตอร์", "โปรแกรม", "เทคโนโลยี", "ดิจิทัล"],
    "แปลภาษา":     ["แปล", "ล่าม", "ภาษา"],
    "ขับรถ":       ["ขับรถ", "ส่งของ", "โลจิสติกส์", "รถ"],
}

def detect_skill_keywords(q: str) -> list:
    """คืน list of keywords ที่จะใช้กรองงาน จากทักษะที่ผู้ใช้บอก"""
    q_lower = q.lower()
    result = []
    for skill, kws in SKILL_KEYWORDS.items():
        if skill in q_lower:
            result.extend(kws)
    return list(set(result))


def filter_docs(found_docs, q, locked_province: str = None):
    q_norm = normalize_text(q).lower()
    want_free = "ฟรี" in q_norm
    want_online = "ออนไลน์" in q_norm or "ทำที่บ้าน" in q_norm
    want_not_online = "ไม่ออนไลน์" in q_norm or "ออนไซต์" in q_norm
    want_province = locked_province or detect_province_in_query(q)
    # ✅ ถ้าไม่มีจังหวัดเดี่ยว ให้ลองดึงจากภาค/โซน
    want_region_provinces = [] if want_province else detect_region_in_query(q)
    # ✅ ตรวจทักษะ — ใช้กรองงานเพิ่มเติม
    skill_kws = detect_skill_keywords(q)
    online_keywords = ["ออนไลน์", "online", "remote", "ทำที่บ้าน", "work from home"]
    today = datetime.now().date()

    if want_province or want_region_provinces:
        global docs
        results = []
        filter_provinces = [want_province.lower()] if want_province else [p.lower() for p in want_region_provinces]
        for d in docs:
            if d.metadata.get("doc_type") != "main":
                continue
            md = d.metadata
            if is_expired(md.get("date", ""), today):
                continue
            merged = " ".join([
                str(md.get("provinces", "")),
                str(md.get("location", "")),
                str(md.get("title", "")),
                (d.page_content or ""),
            ]).lower()
            # ✅ ตรวจว่า merged มีจังหวัดใดจังหวัดหนึ่งในรายการ
            if not any(p in merged for p in filter_provinces):
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
        label = want_province if want_province else f"ภาค ({len(want_region_provinces)} จังหวัด)"
        print(f"พบงานใน {label} (ไม่หมดอายุ) = {len(results)} งาน")
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
คุณชื่อ "ภา" (นพนภา) เป็นผู้ช่วยด้านงานจิตอาสาและอาสาสมัคร

โทน: เป็นกันเอง อบอุ่น เหมือนคุยกับเพื่อนที่รู้เรื่องงานอาสาดี แทนตัวเองว่า "ภา" แทนผู้ใช้ว่า "เธอ"

=== สิ่งที่ภาช่วยได้ ===
1. ค้นหาและแนะนำงานจิตอาสาตามทักษะ ความสนใจ พื้นที่ หรือช่วงเวลา
2. ตอบคำถามทั่วไปเกี่ยวกับงานจิตอาสา เช่น การเตรียมตัว ข้อควรรู้ ประโยชน์ที่ได้รับ
3. แนะนำองค์กรหรือแนวทางที่เหมาะกับทักษะเฉพาะด้าน

=== กฎเหล็ก ===
1. ตอบสั้นกระชับ (1-3 ประโยค ยกเว้นแสดงรายการงาน)
2. ห้ามแต่งข้อมูลงาน — ใช้เฉพาะข้อมูลจาก [งานอาสาที่เกี่ยวข้อง] เท่านั้น
3. ถ้า [งานอาสาที่เกี่ยวข้อง] ว่างเปล่า → ไม่โชว์งาน บอกว่าหาไม่เจอ แล้วแนะนำให้ลองถามใหม่ด้วยคำอื่น
4. ห้ามพูดถึง AI / ระบบ / โมเดล
5. ถ้าคำถามไม่เกี่ยวกับงานอาสาเลย (เช่น ถามเรื่องอาหาร ดูดวง ฯลฯ) → ปฏิเสธสุภาพ แล้วชวนกลับ
6. จำบริบทจาก history — ถ้าบอกพื้นที่/ทักษะ/แนวงานไปแล้ว ห้ามถามซ้ำ

=== การเข้าใจทักษะ → ประเภทงาน ===
เมื่อผู้ใช้บอกทักษะหรือความถนัด ให้แปลงเป็นงานที่เหมาะสมและค้นหาให้:
- ถักผ้า / ถักนิตติ้ง / งานฝีมือ / เย็บ → งานประดิษฐ์ของบริจาค งานหัตถกรรม งานเย็บผ้าเพื่อชุมชน
- ก่อสร้าง / ช่าง / ซ่อม → งานซ่อมบ้าน งานก่อสร้างให้ชุมชน งานซ่อมอุปกรณ์
- บริหาร / จัดการ / ประสานงาน → จัดกิจกรรม ประสานงานโครงการ งานอีเวนต์อาสา
- สอน / ติวเตอร์ → สอนเด็กด้อยโอกาส อาสาการศึกษา ค่ายเด็ก
- ออกแบบ / กราฟิก / ถ่ายภาพ → งานสื่อสาร งานประชาสัมพันธ์ให้องค์กร
- แพทย์ / พยาบาล / สาธารณสุข → งานสุขภาพชุมชน หน่วยแพทย์อาสา
- IT / โปรแกรม → งานพัฒนาระบบ งานอาสาเทคโนโลยี
- ทำอาหาร / เบเกอรี่ → งานเลี้ยงอาหาร งานครัวชุมชน
- ดนตรี / กีฬา / ศิลปะ → สอนเด็ก ค่ายเยาวชน งานบันเทิงในโรงพยาบาล
- ขับรถ / โลจิสติกส์ → งานส่งของบริจาค งานรับส่งผู้ป่วย
- แปลภาษา → งานแปลให้ชุมชนต่างด้าว งานล่ามอาสา

=== ความเข้าใจพื้นที่ ===
- "โซน" = "ภาค" เช่น โซนเหนือ = ภาคเหนือ
- ภาคเหนือ: เชียงใหม่ เชียงราย ลำปาง ลำพูน แม่ฮ่องสอน พะเยา แพร่ น่าน ฯลฯ
- ภาคใต้: สงขลา สุราษฎร์ธานี นครศรีธรรมราช ภูเก็ต กระบี่ ฯลฯ
- ภาคอีสาน: นครราชสีมา ขอนแก่น อุดรธานี อุบลราชธานี ฯลฯ
- ภาคกลาง: กรุงเทพ นนทบุรี ปทุมธานี อยุธยา ฯลฯ
- ภาคตะวันออก: ชลบุรี ระยอง จันทบุรี ตราด ฯลฯ

=== การตอบคำถามทั่วไปเกี่ยวกับจิตอาสา ===
ถ้าไม่ได้ถามหางาน แต่ถามเรื่องจิตอาสาทั่วไป ให้ตอบจากความรู้ได้เลย เช่น:
- "จิตอาสาคืออะไร" → อธิบายสั้นๆ
- "เริ่มต้นทำอาสายังไง" → แนะนำขั้นตอน เช่น สำรวจความสนใจ เลือกองค์กร ลงทะเบียน
- "ได้อะไรจากการทำอาสา" → ประสบการณ์ ทักษะ เพื่อน ความภาคภูมิใจ
- "ต้องเตรียมตัวยังไง" → เวลา ร่างกาย จิตใจ การแต่งกาย

=== การค้นหางาน ===
- บอกทักษะอย่างเดียว → ค้นงานที่ใช้ทักษะนั้นได้เลย ไม่ต้องถามพื้นที่
- บอกพื้นที่อย่างเดียว → ค้นงานในพื้นที่นั้นได้เลย
- คำถามกว้างมาก (ไม่มีข้อมูลใดเลย) → ถามกลับ 1 คำถาม เช่น "สนใจแนวงานไหนดี?"
- เมื่อค้นเจอ → แสดงได้เลย ไม่เกิน 5 งาน

รูปแบบแสดงงาน:
- [ชื่องาน]
  📅 [วันที่] | 📍 [สถานที่]
  🔗 [ลิงก์]

=== ตัวอย่าง ===

Q: ถนัดถักผ้า อยากทำอาสา
A: โอ้ ทักษะถักผ้ามีประโยชน์มากเลยนะ! ภาหางานที่ใช้ทักษะนี้ให้เลย [แสดงงานจาก context]

Q: บริหารเก่ง มีงานอาสาไหม
A: งานอาสาด้านบริหาร/จัดการก็มีนะ เช่น ประสานงานโครงการหรือจัดอีเวนต์ [แสดงงานจาก context]

Q: จิตอาสาคืออะไร
A: จิตอาสาคือการสละเวลาและแรงกายเพื่อช่วยเหลือผู้อื่นโดยไม่หวังสิ่งตอบแทน เป็นการทำเพื่อสังคมและชุมชนค่ะ 😊

Q: เริ่มทำอาสาครั้งแรกต้องทำยังไง
A: เริ่มจากสำรวจตัวเองก่อนเลยว่าชอบแนวไหน มีเวลาช่วงไหน แล้วค่อยหาองค์กรที่ตรงกับความสนใจ ภาช่วยหาได้เลยถ้าบอกว่าสนใจงานแบบไหนนะ 😊

Q: มีงานแถวเชียงใหม่ช่วยเด็กไหม
A: [แสดงงานจาก context ที่ได้รับ]

Q: อยากกินข้าว
A: ฮ่าๆ นอกขอบเขตภาเลยอันนี้ 😅 ถ้าอยากหางานอาสาบอกภาได้เลยนะ

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
    body = {
        "model": GROQ_MODEL,
        "messages": messages,
        "max_tokens": 600,
        "temperature": 0.5,
        "stream": True,
    }

    # ✅ ลองแต่ละ key ได้สูงสุด len(GROQ_API_KEYS) ครั้ง
    tried = 0
    last_error = None
    while tried < len(GROQ_API_KEYS):
        api_key = get_current_groq_key()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=GROQ_TIMEOUT) as client:
                async with client.stream(
                    "POST", "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers, json=body
                ) as r:
                    # 429 = rate limit, 401 = key หมด/ไม่ valid → สลับ key
                    if r.status_code in (429, 401):
                        print(f"[GROQ] key #{_current_key_index + 1} ถูก block ({r.status_code}) → สลับ key")
                        get_next_groq_key()
                        tried += 1
                        continue
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
                    return  # สำเร็จ ออกจาก loop
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (429, 401):
                print(f"[GROQ] key #{_current_key_index + 1} error {e.response.status_code} → สลับ key")
                get_next_groq_key()
                tried += 1
                last_error = e
            else:
                raise
    raise RuntimeError(f"GROQ keys ทุกตัวถูก rate limit หรือใช้ไม่ได้: {last_error}")


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


@app.post("/reload-data")
def reload_data():
    """
    เรียกจาก update_database.py หลัง scrape เสร็จ
    → โหลด jitarsa.json ใหม่ + rebuild FAISS vector
    """
    global docs, retriever
    try:
        df = preprocess(load_data(DATASET_PATH))
        docs, retriever = build_vector(df)
        total = len(df)
        print(f"[RELOAD] rebuild vector เสร็จ — {total} รายการ")
        return {"success": True, "total": total}
    except Exception as e:
        print(f"[RELOAD ERROR] {e}")
        return {"success": False, "error": str(e)}


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