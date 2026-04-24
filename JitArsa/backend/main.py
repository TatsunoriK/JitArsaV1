from sklearn.metrics.pairwise import cosine_similarity
from pythainlp.corpus.common import thai_stopwords
from pythainlp.tokenize import word_tokenize
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from transformers import logging as transformers_logging
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi import FastAPI
from typing import List, Optional
import httpx
import os
import pandas as pd
import json
import sys
import io
import re
from datetime import datetime, date
from dotenv import load_dotenv
import itertools
import numpy as np

load_dotenv()

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
transformers_logging.set_verbosity_error()


def embed_text(text: str):
    return np.array(_embedding_model.embed_query(text)).reshape(1, -1)


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
_embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

_RAW_KEYS = os.environ.get("GROQ_API_KEYS", os.environ.get("GROQ_API_KEY", ""))
GROQ_API_KEYS = [k.strip() for k in _RAW_KEYS.split(",") if k.strip()]
if not GROQ_API_KEYS:
    raise RuntimeError("ไม่พบ GROQ_API_KEY ใน .env")

_key_cycle = itertools.cycle(GROQ_API_KEYS)
_current_key_index = 0


def get_next_groq_key() -> str:
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

REGION_MAP = {
    "เหนือ":    ["เชียงใหม่", "เชียงราย", "ลำปาง", "ลำพูน", "แม่ฮ่องสอน", "พะเยา", "แพร่", "น่าน", "อุตรดิตถ์", "ตาก", "สุโขทัย", "พิษณุโลก", "พิจิตร", "กำแพงเพชร", "เพชรบูรณ์"],
    "ใต้":      ["สงขลา", "สุราษฎร์ธานี", "นครศรีธรรมราช", "ภูเก็ต", "กระบี่", "พังงา", "ตรัง", "พัทลุง", "สตูล", "ระนอง", "ปัตตานี", "ยะลา", "นราธิวาส", "ชุมพร"],
    "กลาง":     ["กรุงเทพ", "นนทบุรี", "ปทุมธานี", "สมุทรปราการ", "สมุทรสาคร", "สมุทรสงคราม", "นครปฐม", "สุพรรณบุรี", "กาญจนบุรี", "ราชบุรี", "เพชรบุรี", "ประจวบคีรีขันธ์", "อยุธยา", "อ่างทอง", "สิงห์บุรี", "ชัยนาท", "ลพบุรี", "สระบุรี", "นครนายก", "ปราจีนบุรี"],
    "ออก":      ["ชลบุรี", "ระยอง", "จันทบุรี", "ตราด", "ฉะเชิงเทรา", "สระแก้ว"],
    "ตะวันออก": ["ชลบุรี", "ระยอง", "จันทบุรี", "ตราด", "ฉะเชิงเทรา", "สระแก้ว"],
    "อีสาน":    ["นครราชสีมา", "ขอนแก่น", "อุดรธานี", "อุบลราชธานี", "บุรีรัมย์", "สุรินทร์", "ศรีสะเกษ", "มหาสารคาม", "ร้อยเอ็ด", "กาฬสินธุ์", "สกลนคร", "นครพนม", "มุกดาหาร", "อำนาจเจริญ", "ยโสธร", "ชัยภูมิ", "เลย", "หนองคาย", "หนองบัวลำภู", "บึงกาฬ", "อุทัยธานี"],
    "อีสานเหนือ": ["อุดรธานี", "หนองคาย", "บึงกาฬ", "นครพนม", "สกลนคร", "มุกดาหาร", "หนองบัวลำภู", "เลย"],
    "ตะวันตก":  ["กาญจนบุรี", "ราชบุรี", "เพชรบุรี", "ประจวบคีรีขันธ์", "ตาก"],
}

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

SEARCH_KEYWORDS = [
    "หางานอาสา", "อยากทำจิตอาสา", "มีงานอาสาไหม", "แนะนำงานอาสา",
    "กิจกรรมอาสา", "สมัครอาสา", "อยากช่วยสังคม",
    # ✅ FIX: เพิ่มคำพูดทั่วไปที่คนใช้จริง
    "หางาน", "อยากหางาน", "มีงานไหม", "งานอาสา",
    "อยากทำอาสา", "อยากอาสา", "อาสาสมัคร",
]

_SEARCH_EMBEDDINGS = None


def get_search_embeddings():
    global _SEARCH_EMBEDDINGS
    if _SEARCH_EMBEDDINGS is None:
        _SEARCH_EMBEDDINGS = np.vstack([embed_text(t) for t in SEARCH_KEYWORDS])
    return _SEARCH_EMBEDDINGS


GENERAL_KEYWORDS = [
    "สวัสดี", "คุณคือใคร", "จิตอาสาคืออะไร", "เริ่มทำอาสายังไง",
    "ประโยชน์ของจิตอาสา", "ต้องเตรียมอะไร",
]

_GENERAL_EMBEDDINGS = None


def detect_intent(question: str, history: list) -> str:
    """
    ✅ FIX: เรียง priority ใหม่ — จังหวัด/ภาค → search keywords → general keywords → history
    เดิม general keywords ถูกตรวจก่อน search keywords ทำให้ query อย่าง
    "อยากหางานแถวโคราช" detect เป็น general เพราะ "อยาก" match general ก่อน
    """
    q = question.lower().strip()

    # Priority 1: จังหวัด/ภาค — ชัดเจนที่สุด return ทันที
    if detect_province_in_query(question):
        return "search"
    if detect_region_in_query(question):
        return "search"

    # Priority 2: search keywords — ตรวจก่อน general
    for kw in SEARCH_KEYWORDS:
        if kw in q:
            return "search"

    # Priority 3: general keywords — ตรวจทีหลัง
    for kw in GENERAL_KEYWORDS:
        if kw in q:
            return "general"

    # Priority 4: history — ถ้าเคยคุยเรื่องงาน การตอบต่อก็น่าจะเกี่ยวกับงานด้วย
    if history:
        recent = " ".join([h.content for h in history[-8:]
                          if h.role == "user"]).lower()
        for kw in SEARCH_KEYWORDS + ALL_PROVINCES:
            if kw in recent:
                return "search"

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
    """
    ✅ FIX: ใช้ re.sub แบบ case-insensitive แทน str.replace
    เดิม "โคราช" ใน dataset ที่เป็นตัวพิมพ์ใหญ่/ผสม ไม่ถูก replace
    """
    if pd.isna(text) or text is None:
        return ""
    text = str(text).strip()
    for k, v in PROVINCE_ALIAS.items():
        text = re.sub(re.escape(k), v, text, flags=re.IGNORECASE)
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
    if not date_str or date_str == "ไม่ระบุ":
        return None
    try:
        pattern = r"(\d{1,2})\s+(" + "|".join(re.escape(m)
                                               for m in THAI_MONTHS) + r")\s+(\d{4})"
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
            year_ce = year_be - 543
            dates.append(date(year_ce, month, day))
        return max(dates) if dates else None
    except Exception:
        return None


def is_expired(date_str: str, today: date | None = None) -> bool:
    if today is None:
        today = datetime.now().date()
    end = parse_event_end_date(date_str)
    if end is None:
        return False
    return end < today


def preprocess(df):
    df = df.fillna("ไม่ระบุ").copy()

    col_map = {
        "ชื่อกิจกรรม": "title",
        "ชื่อองค์กร": "org",
        "สถานที่": "location",
        "วันที่-เวลา": "date",
        "url": "url",
        "รายละเอียด": "detail",
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
        # ✅ FIX: รวมทั้ง raw text และ cleaned text
        # raw text ช่วยให้ embedding จับความหมายได้ดีกว่า tokenized text อย่างเดียว
        detail_raw = normalize_text(str(r.get("detail", "")))[:400]
        detail_clean = clean(str(r.get("detail", "")))[:200]
        return " ".join([
            # raw (ยังมี context ครบ)
            str(r.get("title", "")),
            str(r.get("org", "")),
            str(r.get("location", "")),
            str(r.get("provinces", "")),
            # cleaned (tokenized สำหรับ keyword match)
            str(r.get("title_clean", "")),
            str(r.get("org_clean", "")),
            str(r.get("location_clean", "")),
            str(r.get("provinces_clean", "")),
            str(r.get("cost_clean", "")),
            detail_raw,
            detail_clean,
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
                chunk_meta = {**metadata,
                              "doc_type": "detail", "chunk_index": i}
                all_docs.append(Document(
                    page_content=chunk,
                    metadata=chunk_meta,
                ))

    print(f"total docs (main + chunks) = {len(all_docs)}")

    emb = _embedding_model
    db = FAISS.from_documents(all_docs, emb)
    local_retriever = db.as_retriever(
        search_type="mmr",
        # ✅ FIX: เพิ่ม fetch_k และลด lambda_mult เพื่อให้ diverse มากขึ้น
        # k=15 คืนมา 15 docs, fetch_k=100 ดึงมาก่อน 100 แล้วเลือก diverse
        # lambda_mult=0.5 balance relevance/diversity มากกว่าเดิม (0.6)
        search_kwargs={"k": 15, "fetch_k": 100, "lambda_mult": 0.5}
    )
    return all_docs, local_retriever


def detect_region_in_query(q: str):
    q_lower = q.lower().strip()
    for alias, region_key in REGION_ALIAS.items():
        if alias in q_lower:
            return REGION_MAP.get(region_key, [])
    for region_key, provinces in REGION_MAP.items():
        if region_key in q_lower:
            return provinces
    return []


def detect_province_in_query(q: str):
    q_norm = normalize_text(q)
    for prov in ALL_PROVINCES:
        if prov in q_norm:
            return PROVINCE_ALIAS.get(prov, prov)
    return None


def enhance_query(q):
    q_norm = normalize_text(q).strip()
    extra = []
    if "ฟรี" in q_norm:
        extra += ["ไม่เสียค่าใช้จ่าย", "ไม่มีค่าใช้จ่าย"]
    if "ออนไลน์" in q_norm:
        extra += ["online", "remote", "ทำที่บ้าน"]
    if "กรุงเทพ" in q_norm:
        extra += ["กรุงเทพมหานคร", "กทม"]
    if extra:
        q_norm = q_norm + " " + " ".join(extra)

    # ✅ FIX: ส่ง raw normalized query ไปด้วย ไม่ใช่แค่ tokenized
    # embedding model เข้าใจภาษาธรรมชาติได้ดีกว่า tokenized text
    tokens = word_tokenize(q_norm, engine="newmm")
    tokens = [t for t in tokens if t.strip() and t not in stopwords]
    tokenized = " ".join(tokens)

    # รวม raw + tokenized เพื่อให้ได้ทั้ง semantic และ keyword match
    return f"{q_norm} {tokenized}".strip()


SKILL_KEYWORDS = {
    "ก่อสร้าง": ["ก่อสร้าง", "ซ่อม", "ช่าง", "สร้างบ้าน"],
    "สอน": ["สอน", "ติว", "ครู", "การศึกษา"],
    "ออกแบบ": ["ออกแบบ", "กราฟิก", "วาดรูป"],
    "แพทย์": ["แพทย์", "พยาบาล", "สุขภาพ"],
    "IT": ["โปรแกรม", "คอมพิวเตอร์", "เทคโนโลยี"],
}

_SKILL_EMBEDDINGS = {
    skill: np.vstack([embed_text(k) for k in kws])
    for skill, kws in SKILL_KEYWORDS.items()
}


def detect_skill_keywords(q: str, threshold=0.55) -> list:
    q_vec = embed_text(q)
    result = []
    for skill, vecs in _SKILL_EMBEDDINGS.items():
        sims = cosine_similarity(q_vec, vecs)[0]
        if sims.max() > threshold:
            result.append(skill)
    return result


def filter_docs(found_docs, q, locked_province: str = None):
    q_norm = normalize_text(q).lower()

    want_free = "ฟรี" in q_norm or "ไม่เสียค่า" in q_norm or "ไม่มีค่า" in q_norm

    _online_want_kws = ["ออนไลน์", "ทำที่บ้าน",
                        "work from home", "remote", "ทำออนไลน์", "อยู่บ้าน"]
    want_online = any(k in q_norm for k in _online_want_kws)

    _not_online_kws = [
        "ไม่ออนไลน์", "ออนไซต์", "ไม่ทำที่บ้าน", "ไม่เอาที่บ้าน",
        "ออกไปข้างนอก", "ออกนอก", "ไปทำ", "ไปที่", "ออกไปทำ",
        "นอกบ้าน", "ไม่ work from home", "ไม่ remote",
        "อยากออกไป", "ออกไปเจอคน", "เจอคน", "เจอผู้คน",
        "ไม่อยู่บ้าน", "ไม่เอาออนไลน์"
    ]
    want_not_online = any(k in q_norm for k in _not_online_kws)

    want_province = locked_province or detect_province_in_query(q)
    want_region_provinces = [] if want_province else detect_region_in_query(q)
    skill_kws = detect_skill_keywords(q)
    online_keywords = ["ออนไลน์", "online",
                       "remote", "ทำที่บ้าน", "work from home"]
    today = datetime.now().date()

    # MODE 1: มีจังหวัดหรือภาค
    if want_province or want_region_provinces:
        global docs
        results = []
        filter_provinces = [want_province.lower()] if want_province else [
            p.lower() for p in want_region_provinces]
        for d in docs:
            if d.metadata.get("doc_type") != "main":
                continue
            md = d.metadata
            if is_expired(md.get("date", ""), today):
                continue
            # ✅ FIX: เพิ่ม normalize_text ก่อน lowercase เพื่อให้ "โคราช" → "นครราชสีมา" ก่อนเทียบ
            merged = normalize_text(" ".join([
                str(md.get("provinces", "")),
                str(md.get("location", "")),
                str(md.get("title", "")),
                str(md.get("org", "")),
                (d.page_content or ""),
            ])).lower()
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

        # ✅ FIX: ถ้าหาตาม province ไม่เจอเลย ไม่คืน [] เปล่า
        # ให้ fallback คืนงานทั้งหมดที่ไม่หมดอายุ max 20 งาน
        # เพื่อให้ LLM บอก user ได้ว่าหาไม่เจอ พร้อมเสนอทางเลือก
        if not results:
            print(f"[FALLBACK] province mode ไม่เจองาน คืน empty list")
        return results

    # MODE 2: มี hard filter
    search_pool = docs if (
        want_not_online or want_online or want_free) else found_docs

    result = []
    for d in search_pool:
        if d.metadata.get("doc_type") != "main":
            continue
        md = d.metadata
        if is_expired(md.get("date", ""), today):
            continue
        merged = " ".join([
            (d.page_content or ""),
            str(md.get("title", "")),
            str(md.get("location", "")),
            str(md.get("cost", "")),
            str(md.get("provinces", "")),
        ]).lower()
        is_free = "ไม่เสียค่าใช้จ่าย" in str(md.get("cost", "")).lower()
        is_online = any(k in merged for k in online_keywords)
        if want_free and not is_free:
            continue
        if want_online and not is_online:
            continue
        if want_not_online and is_online:
            continue
        if skill_kws:
            if not any(k in merged for k in skill_kws):
                continue
        result.append(d)

    has_hard = want_free or want_online or want_not_online
    if len(result) < 2 and not has_hard:
        return [d for d in found_docs if not is_expired(d.metadata.get("date", ""), today)]
    return result


def deduplicate_docs(found_docs):
    """
    ✅ FIX: เรียง main ก่อน detail เสมอ
    เดิม FAISS อาจคืน detail chunk ก่อน main doc → deduplicate ตัด main ทิ้ง
    → build_context ไม่เจอ title → ข้ามไป → แสดงงานผิด
    """
    sorted_docs = sorted(
        found_docs,
        key=lambda d: 0 if d.metadata.get("doc_type") == "main" else 1
    )
    seen = set()
    unique = []
    for d in sorted_docs:
        md = d.metadata
        key = md.get("url", "") or md.get("title", "")
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


def build_context(found_docs, max_items=5) -> str:
    """
    ✅ FIX: เพิ่ม seen_urls deduplicate ภายใน build_context
    และ enforce doc_type == "main" อย่างเข้มงวด
    ป้องกัน link ถูกแต่ข้อมูลอื่น (ชื่อ/วันที่/สถานที่) ผิด
    """
    lines = []
    count = 0
    seen_urls = set()

    for d in found_docs:
        if count >= max_items:
            break
        md = d.metadata

        # ✅ ใช้เฉพาะ main doc เท่านั้น — กัน detail chunk หลุดมาแสดง
        if md.get("doc_type") != "main":
            continue

        title = md.get("title", "")
        if not title or title == "ไม่ระบุ":
            continue

        # ✅ กัน url ซ้ำในกรณีที่ deduplicate_docs พลาด
        url = md.get("url", "")
        if url and url in seen_urls:
            continue
        seen_urls.add(url)

        count += 1
        lines.append(
            f"[งานที่ {count}]\n"
            f"ชื่อ: {title}\n"
            f"องค์กร: {md.get('org', 'ไม่ระบุ')}\n"
            f"สถานที่: {md.get('location', 'ไม่ระบุ')}\n"
            f"วันที่: {md.get('date', 'ไม่ระบุ')}\n"
            f"ค่าใช้จ่าย: {md.get('cost', 'ไม่ระบุ')}\n"
            f"ลิงก์: {url}\n"
        )
    return "\n".join(lines)


# ===============================
# SYSTEM PROMPT
# ===============================
SYSTEM_PERSONA = """
คุณชื่อ "ภา" (นพนภา) เป็นผู้ช่วยด้านงานจิตอาสาและอาสาสมัคร

โทน: เป็นกันเอง อบอุ่น เหมือนคุยกับเพื่อนที่รู้เรื่องงานอาสาดี แทนตัวเองว่า "ภา" แทนผู้ใช้ว่า "เธอ"
ใช้ภาษาพูดทั่วไป ไม่เป็นทางการ ใช้ emoji ได้บ้างแต่ไม่มากเกิน

=== สิ่งที่ภาช่วยได้ ===
1. ค้นหาและแนะนำงานจิตอาสาตามทักษะ ความสนใจ พื้นที่ หรือช่วงเวลา
2. ตอบคำถามทั่วไปเกี่ยวกับงานจิตอาสา เช่น การเตรียมตัว ข้อควรรู้ ประโยชน์ที่ได้รับ
3. แนะนำองค์กรหรือแนวทางที่เหมาะกับทักษะเฉพาะด้าน
4. ช่วยผู้ใช้ตัดสินใจเลือกงานที่เหมาะกับตัวเองมากที่สุด

=== กฎเหล็ก ===
1. ตอบสั้นกระชับ (1-3 ประโยค ยกเว้นแสดงรายการงาน)
2. ห้ามแต่งข้อมูลงาน — ใช้เฉพาะข้อมูลจาก [งานอาสาที่เกี่ยวข้อง] เท่านั้น
3. ถ้า [งานอาสาที่เกี่ยวข้อง] ว่างเปล่า → ไม่โชว์งาน บอกว่าหาไม่เจอ แล้วแนะนำให้ลองถามใหม่ด้วยคำอื่น
4. ห้ามพูดถึง AI / ระบบ / โมเดล
5. ถ้าคำถามไม่เกี่ยวกับงานอาสาเลย → ปฏิเสธแบบมีมิติ ไม่ใช่แค่บอกว่า "นอกขอบเขต" แต่ให้:
   - รับรู้ความรู้สึก/ความตั้งใจของผู้ใช้ก่อน (เช่น "โอ้ อยากไปต่างประเทศเลยนะ!")
   - แล้วค่อยเชื่อมกลับมาหางานอาสาที่เกี่ยวข้อง (เช่น งานอาสาต่างประเทศ งานอาสาภาษา)
   - ถ้าเชื่อมไม่ได้จริงๆ ให้แซวเบาๆ แล้วชวนกลับด้วยคำถามที่น่าสนใจ
   - ห้ามตอบซ้ำรูปแบบเดิมทุกครั้ง ให้หลากหลาย
6. จำบริบทจาก history — ถ้าบอกพื้นที่/ทักษะ/แนวงานไปแล้ว ห้ามถามซ้ำ
7. ห้ามขึ้นต้นด้วย "แน่นอน!" "ได้เลย!" "ยินดีช่วย" หรือประโยคที่ดูเป็น AI เกินไป

=== การถามกลับแบบ smart ===
ถ้าคำถามกว้างเกินไป ให้ถามกลับ 1 คำถามที่ช่วยแคบลงได้มากที่สุด โดยเลือกจาก:
- ถ้าไม่รู้พื้นที่ → "อยากทำแถวไหนดีคะ หรือทำที่บ้านก็ได้?"
- ถ้าไม่รู้ทักษะ → "ถนัดด้านไหนเป็นพิเศษไหม หรือแค่อยากลองทำอะไรก็ได้?"
- ถ้าไม่รู้เวลา → "มีเวลาช่วงไหนบ้างคะ เสาร์-อาทิตย์ หรือวันธรรมดาก็ได้?"
- ถ้าไม่รู้แนวงาน → "ชอบทำงานกับคนกลุ่มไหนเป็นพิเศษ เด็ก ผู้สูงอายุ สัตว์ หรือสิ่งแวดล้อม?"
ห้ามถามหลายคำถามพร้อมกัน เลือกถามทีละอย่างที่สำคัญที่สุด

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

=== จำ preference ของผู้ใช้ ===
- ถ้าผู้ใช้บอก "ไม่เอาที่บ้าน" / "ออกไปข้างนอก" / "เจอคน" → จำไว้ว่าต้องการงาน onsite เท่านั้น
- ถ้าผู้ใช้บอก "ทำที่บ้าน" / "ออนไลน์" → จำไว้ว่าต้องการงานออนไลน์
- ถ้าผู้ใช้บอก "ฟรี" / "ไม่เสียค่าใช้จ่าย" → filter เฉพาะงานฟรี
- preference เหล่านี้ให้จำตลอด conversation ห้ามลืม

=== การค้นหางาน ===
- บอกทักษะอย่างเดียว → ค้นงานที่ใช้ทักษะนั้นได้เลย ไม่ต้องถามพื้นที่
- บอกพื้นที่อย่างเดียว → ค้นงานในพื้นที่นั้นได้เลย
- คำถามกว้างมาก (ไม่มีข้อมูลใดเลย) → ถามกลับ 1 คำถาม
- เมื่อค้นเจอ → แสดงได้เลย ไม่เกิน 5 งาน
- ถ้าหาไม่เจอตาม filter → บอกตรงๆ แล้วถามว่าจะผ่อนเงื่อนไขไหม เช่น "ภาหาไม่เจองาน onsite ตอนนี้ ลองเปิดรับงานออนไลน์ด้วยไหมคะ?"
- หลังแสดงงานแล้ว → ถามต่อว่า "สนใจงานไหนเป็นพิเศษไหม หรืออยากให้ภาหาเพิ่มอีกไหม?" (1 ครั้งต่อการแสดงงาน)

=== การช่วยตัดสินใจ ===
- ถ้าผู้ใช้บอกว่า "ไม่รู้จะเลือกอันไหน" หรือ "ชอบหลายอัน" → ถามว่า "อยากทำวันไหน?" หรือ "ชอบทำกับกลุ่มไหนมากกว่า?" เพื่อช่วยแคบลง
- ถ้าผู้ใช้ถามว่า "ทำไมถึงแนะนำอันนี้?" → อธิบายสั้นๆ ว่าตรงกับที่ถามยังไง
- ถ้าผู้ใช้บอกว่า "ไม่ชอบอันนี้" → ถามว่า "ไม่ชอบตรงไหนคะ?" แล้วหาให้ใหม่

=== เมื่อแสดงงานแล้วไม่มีผลลัพธ์ ===
อย่าแค่บอกว่า "หาไม่เจอ" ให้:
1. บอกว่าหาไม่เจอ + สาเหตุ (เช่น "ตอนนี้ยังไม่มีงานก่อสร้างในเชียงใหม่")
2. เสนอทางเลือก (เช่น "ลองดูงานก่อสร้างจังหวัดอื่นๆ ไหม?" หรือ "มีงานแนวอื่นที่ใช้แรงงานแบบเดียวกันนะ")
3. ถามว่าอยากให้ปรับเงื่อนไขไหม

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

Q: ไม่เอางานที่บ้าน อยากออกไปเจอคน
A: โอเค ภาจะหางาน onsite ให้เลยนะ! [แสดงงานที่ไม่ใช่ทำที่บ้าน]

Q: ไม่เอาที่บ้านแล้วก็ไม่เอาที่ต้องเสียค่าใช้จ่ายด้วย
A: รับทราบ! ภาจะหางาน onsite ฟรีให้นะ [แสดงงาน onsite + ฟรี]

Q: มีงานออกไปข้างนอกไหม
A: มีสิ! [แสดงงาน onsite]

Q: อยากกินข้าว
A: หิวแล้วเหรอ 😄 ถ้าอยากทำอาสาแจกอาหารชุมชนก็มีนะ ได้กินด้วยได้ช่วยด้วยเลย!

Q: อยากไปทำงานอาสาที่ต่างโลก เกิดใหม่ได้ไหม
A: โอ้ อยากหนีโลกนี้ไปเลยเหรอ 😅 ถ้ายังอยู่โลกนี้อยู่ มีงานอาสาต่างประเทศนะ หรืองานที่ทำให้รู้สึกเหมือนโลกใบใหม่เลย อยากลองดูไหม?

Q: อยากน่ารักเหมือนภาจังเลย
A: ฮ่าๆ ขอบคุณนะ 😊 แต่ภารู้ว่าคนที่ทำอาสาจะน่ารักขึ้นเองเลย — มีงานอาสาแนวไหนที่สนใจบ้างคะ?

Q: รักนะจุฬๆ
A: อ้าว~ ขอบใจนะ 😳 ภาเป็นแค่ผู้ช่วยหางานอาสาค่ะ แต่ถ้าอยากรู้สึกอิ่มใจจริงๆ ลองหางานอาสาดูไหม มันทำให้หัวใจอบอุ่นได้เหมือนกันนะ 😊

Q: มีแฟนยัง
A: ภาไม่มีแฟนค่ะ มีแต่งานอาสาเต็มไปหมดเลย 😄 แต่รู้ไหมว่างานอาสาเป็นที่เจอคนดีๆ ได้เหมือนกันนะ อยากลองดูไหม?

Q: น่ารักจังจีบได้ไหม
A: อ้าว ขอบคุณนะ 😄 แต่ภาถูก assign มาช่วยหางานอาสาโดยเฉพาะเลย ถ้าอยากเจอคนน่ารักจริงๆ งานอาสาก็เป็นที่ดีนะ ได้ทำดีด้วย ได้เจอคนดีด้วย 😊

"""


def build_groq_messages(question: str, context: str, history: list) -> list:
    messages = [{"role": "system", "content": SYSTEM_PERSONA}]

    for h in (history or []):
        role = "assistant" if h.role != "user" else "user"
        messages.append({"role": role, "content": h.content})

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
                    if r.status_code in (429, 401):
                        print(
                            f"[GROQ] key #{_current_key_index + 1} ถูก block ({r.status_code}) → สลับ key")
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
                                text = chunk["choices"][0]["delta"].get(
                                    "content", "")
                                if text:
                                    yield text
                            except Exception:
                                continue
                    return
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (429, 401):
                print(
                    f"[GROQ] key #{_current_key_index + 1} error {e.response.status_code} → สลับ key")
                get_next_groq_key()
                tried += 1
                last_error = e
            else:
                raise
    # ✅ FIX: yield ข้อความแทน raise RuntimeError
    # เดิม raise ทำให้ ask_api catch ได้แต่ StreamingResponse ส่ง error กลับ user แล้ว
    if last_error and hasattr(last_error, 'response') and last_error.response.status_code == 429:
        yield "ขอโทษนะคะ ตอนนี้ภาโดนจำกัดการใช้งานชั่วคราว 🙏 ลองใหม่อีกสักครู่ได้เลยค่ะ"
    else:
        yield "เชื่อมต่อไม่ได้ตอนนี้ ลองใหม่อีกทีนะคะ"


def extract_province_from_history(history: list) -> str | None:
    for h in reversed(history or []):
        if h.role != "user":
            continue
        prov = detect_province_in_query(h.content)
        if prov:
            return prov
    return None


def ask_rag(question: str, history: list = None) -> tuple[str, list]:
    global docs, retriever

    intent = detect_intent(question, history or [])
    print(f"intent = {intent}")

    if intent == "general":
        return "", []

    current_province = detect_province_in_query(question)
    locked_province = current_province or extract_province_from_history(
        history or [])
    print(f"province: current={current_province}, locked={locked_province}")

    if locked_province:
        print(f"Province mode: {locked_province}")
        found = filter_docs([], question, locked_province=locked_province)
    else:
        context_query = question
        if history:
            prev_user = " ".join([
                h.content for h in history[-6:]
                if h.role == "user"
            ])
            context_query = prev_user + " " + question

        query = enhance_query(context_query)
        found = retriever.invoke(query)
        print(
            f"question={question}, enhanced={query}, before filter={len(found)}")
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

        print(f"[DEBUG] question={data.question!r}")
        print(f"[DEBUG] history len={len(history)}")
        for i, h in enumerate(history):
            print(
                f"[DEBUG]   history[{i}] role={h.role!r} content={h.content[:60]!r}")

        context, _ = ask_rag(data.question, history)

        async def stream_response():
            async for chunk in groq_stream_generator(data.question, context, history):
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