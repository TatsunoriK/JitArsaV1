# ==============================================================
# main.py — Backend หลักของระบบแชทบอท "ภา" (JitArsa)
#
# ภาพรวมระบบ (RAG Pipeline):
#   1. โหลด dataset งานอาสา (.json) → preprocess → embed → FAISS
#   2. รับคำถามจาก user → detect intent → ค้นหา docs ที่เกี่ยวข้อง
#   3. สร้าง context จาก docs → ส่งให้ Groq LLM → stream คำตอบกลับ
#
# Flow แบบย่อ:
#   POST /ask-pha
#     → detect_intent()          # search หรือ general?
#     → ask_rag()                # ดึง docs ที่เกี่ยวข้อง
#       → filter_docs()          # กรองตาม province/ออนไลน์/ฟรี
#       → deduplicate_docs()     # ลบซ้ำ
#       → build_context()        # สร้าง text context
#     → groq_stream_generator()  # ส่ง context + คำถาม → LLM → stream
# ==============================================================

# ── ML / NLP ──────────────────────────────────────────────────
from sklearn.metrics.pairwise import cosine_similarity   # คำนวณความคล้ายระหว่าง vector
from pythainlp.corpus.common import thai_stopwords       # คำหยุดภาษาไทย เช่น "ที่" "และ"
from pythainlp.tokenize import word_tokenize             # ตัดคำภาษาไทย
from langchain_text_splitters import RecursiveCharacterTextSplitter  # ตัด text ยาวเป็น chunks
from langchain_community.vectorstores import FAISS       # vector database ค้นหาด้วย embedding
from langchain_huggingface import HuggingFaceEmbeddings  # โมเดล embed text → vector
from langchain_core.documents import Document            # wrapper เก็บ text + metadata
from transformers import logging as transformers_logging # ปิด warning จาก HuggingFace

# ── FastAPI ───────────────────────────────────────────────────
from fastapi.responses import StreamingResponse          # ส่ง response แบบ stream ทีละ chunk
from fastapi.middleware.cors import CORSMiddleware       # อนุญาต cross-origin จาก frontend
from pydantic import BaseModel                           # validate request body อัตโนมัติ
from fastapi import FastAPI

# ── Standard Library ──────────────────────────────────────────
from typing import List, Optional
import httpx        # async HTTP client สำหรับเรียก Groq API
import os           # อ่าน env vars และ path
import pandas as pd # จัดการ dataset JSON
import json         # parse SSE response จาก Groq
import sys          # แทนที่ stdout/stderr ให้รองรับ UTF-8
import io           # ใช้คู่กับ sys สำหรับ wrap TextIOWrapper
import re           # regex parse วันที่ภาษาไทย และ alias
from datetime import datetime, date   # เปรียบเทียบวันหมดอายุ
from dotenv import load_dotenv        # โหลด GROQ_API_KEY จากไฟล์ .env
import itertools    # itertools.cycle สำหรับวน API keys
import numpy as np  # คำนวณ cosine similarity

# โหลด .env ก่อนทุกอย่าง เพื่อให้ os.environ อ่าน key ได้
load_dotenv()

# แก้ปัญหา UnicodeEncodeError บน Windows ที่ terminal ไม่รองรับ UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ปิด warning ของ HuggingFace เช่น "Some weights not initialized"
transformers_logging.set_verbosity_error()


def embed_text(text: str):
    """แปลง text → numpy array shape (1, embedding_dim) พร้อมคำนวณ cosine similarity"""
    return np.array(_embedding_model.embed_query(text)).reshape(1, -1)


def safe_print(*args, **kwargs):
    """print ที่กัน UnicodeEncodeError — fallback เป็น ASCII ถ้า encode ไม่ได้"""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        print(text.encode('utf-8', errors='replace').decode('ascii', errors='replace'))


# ================================================================
# 1) FASTAPI — สร้าง app instance และตั้งค่า CORS
# ================================================================
app = FastAPI()

# อนุญาตทุก origin เพื่อให้ Frontend (React/Vite) เรียก API ได้
# allow_credentials=True จำเป็นถ้าต้องการส่ง cookie หรือ Authorization header
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================================================
# 2) CONFIG — ค่าคงที่ทั้งหมด
# ================================================================

# BASE_DIR = โฟลเดอร์ที่ main.py อยู่
# ใช้ build path สัมพัทธ์ → กัน path ผิดเมื่อ cd ไป directory อื่น
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data/jitarsa.json")

# โมเดล embedding รองรับหลายภาษารวม Thai
# ขนาดเล็ก เร็ว ใช้ฟรี ไม่ต้องมี API key
# output: vector 384 มิติ ต่อ 1 ประโยค
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# ── Groq API Key Rotation ──────────────────────────────────────
# รองรับหลาย key คั่นด้วย comma ใน .env เช่น GROQ_API_KEYS=key1,key2,key3
# ถ้าไม่มี GROQ_API_KEYS → fallback ไปหา GROQ_API_KEY (key เดี่ยว)
# ประโยชน์: ถ้า key ไหน rate limit (429) → สลับไป key ถัดไปอัตโนมัติ
_RAW_KEYS = os.environ.get("GROQ_API_KEYS", os.environ.get("GROQ_API_KEY", ""))
GROQ_API_KEYS = [k.strip() for k in _RAW_KEYS.split(",") if k.strip()]
if not GROQ_API_KEYS:
    raise RuntimeError("ไม่พบ GROQ_API_KEY ใน .env")

_key_cycle = itertools.cycle(GROQ_API_KEYS)  # สำรองไว้ ใช้ index-based จริงๆ
_current_key_index = 0                        # index ของ key ที่ใช้อยู่ตอนนี้


def get_next_groq_key() -> str:
    """เลื่อนไปใช้ key ถัดไปในลิสต์ (วนกลับต้นเมื่อถึงตัวสุดท้าย)"""
    global _current_key_index
    _current_key_index = (_current_key_index + 1) % len(GROQ_API_KEYS)
    key = GROQ_API_KEYS[_current_key_index]
    # แสดงแค่ 6 ตัวสุดท้ายของ key เพื่อ debug โดยไม่เปิดเผย key เต็ม
    print(f"[GROQ] สลับไปใช้ key #{_current_key_index + 1} (****{key[-6:]})")
    return key


def get_current_groq_key() -> str:
    """คืน key ที่ใช้อยู่ตอนนี้โดยไม่เลื่อน index"""
    return GROQ_API_KEYS[_current_key_index]


# โมเดล LLM ที่ Groq host ไว้ — ฟรีและเร็ว (แต่มี rate limit)
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_TIMEOUT = 60   # วินาที — ถ้า Groq ไม่ตอบใน 60s ให้ timeout

# ── Text Splitting ────────────────────────────────────────────
# ใช้ตอนแตก "รายละเอียด" งานที่ยาวๆ เป็น chunks เล็กๆ
# เพื่อให้ FAISS ค้นหา keyword ในรายละเอียดได้แม้ text ยาวมาก
CHUNK_SIZE = 500     # ตัวอักษรสูงสุดต่อ chunk
CHUNK_OVERLAP = 100  # overlap ระหว่าง chunk กัน context ขาดตรงรอยต่อ


# ================================================================
# 3) NLP SETUP — ชุดข้อมูลภาษาไทย
# ================================================================

# ลบคำอาสาสำคัญออกจาก stopwords
# เพราะถ้าลบออกจะทำให้ค้นหาคำพวกนี้ไม่เจอ
stopwords = set(thai_stopwords()) - {
    "กิจกรรม", "โครงการ", "สมัคร", "อาสาสมัคร"
}

# แปลงชื่อเรียกต่างๆ ของจังหวัดให้เป็นชื่อมาตรฐาน
# ใช้ตอน normalize dataset และตอนแปลง query ของผู้ใช้
# เช่น user พิมพ์ "โคราช" → normalize เป็น "นครราชสีมา" ก่อน match กับ dataset
PROVINCE_ALIAS = {
    "กทม": "กรุงเทพ",
    "กรุงเทพฯ": "กรุงเทพ",
    "กรุงเทพมหานคร": "กรุงเทพ",
    "bangkok": "กรุงเทพ",
    "bkk": "กรุงเทพ",
    "โคราช": "นครราชสีมา",
    "อยุธยา": "พระนครศรีอยุธยา",
}

# mapping ภาค/โซน → รายชื่อจังหวัดทั้งหมดในภาคนั้น
# ใช้ตอน user พิมพ์ "ภาคเหนือ" หรือ "โซนใต้"
# แทนที่จะระบุจังหวัดตรงๆ → expand เป็น list จังหวัดแล้ว filter
REGION_MAP = {
    "เหนือ":      ["เชียงใหม่", "เชียงราย", "ลำปาง", "ลำพูน", "แม่ฮ่องสอน", "พะเยา", "แพร่", "น่าน", "อุตรดิตถ์", "ตาก", "สุโขทัย", "พิษณุโลก", "พิจิตร", "กำแพงเพชร", "เพชรบูรณ์"],
    "ใต้":        ["สงขลา", "สุราษฎร์ธานี", "นครศรีธรรมราช", "ภูเก็ต", "กระบี่", "พังงา", "ตรัง", "พัทลุง", "สตูล", "ระนอง", "ปัตตานี", "ยะลา", "นราธิวาส", "ชุมพร"],
    "กลาง":       ["กรุงเทพ", "นนทบุรี", "ปทุมธานี", "สมุทรปราการ", "สมุทรสาคร", "สมุทรสงคราม", "นครปฐม", "สุพรรณบุรี", "กาญจนบุรี", "ราชบุรี", "เพชรบุรี", "ประจวบคีรีขันธ์", "อยุธยา", "อ่างทอง", "สิงห์บุรี", "ชัยนาท", "ลพบุรี", "สระบุรี", "นครนายก", "ปราจีนบุรี"],
    "ออก":        ["ชลบุรี", "ระยอง", "จันทบุรี", "ตราด", "ฉะเชิงเทรา", "สระแก้ว"],
    "ตะวันออก":   ["ชลบุรี", "ระยอง", "จันทบุรี", "ตราด", "ฉะเชิงเทรา", "สระแก้ว"],  # alias ของ "ออก"
    "อีสาน":      ["นครราชสีมา", "ขอนแก่น", "อุดรธานี", "อุบลราชธานี", "บุรีรัมย์", "สุรินทร์", "ศรีสะเกษ", "มหาสารคาม", "ร้อยเอ็ด", "กาฬสินธุ์", "สกลนคร", "นครพนม", "มุกดาหาร", "อำนาจเจริญ", "ยโสธร", "ชัยภูมิ", "เลย", "หนองคาย", "หนองบัวลำภู", "บึงกาฬ", "อุทัยธานี"],
    "อีสานเหนือ": ["อุดรธานี", "หนองคาย", "บึงกาฬ", "นครพนม", "สกลนคร", "มุกดาหาร", "หนองบัวลำภู", "เลย"],
    "ตะวันตก":    ["กาญจนบุรี", "ราชบุรี", "เพชรบุรี", "ประจวบคีรีขันธ์", "ตาก"],
}

# แปลงภาษาพูดทั่วไป → key ใน REGION_MAP
# เช่น "ภาคเหนือ" → "เหนือ" → REGION_MAP["เหนือ"] = [จังหวัด...]
REGION_ALIAS = {
    "ภาคเหนือ": "เหนือ", "โซนเหนือ": "เหนือ", "แถบเหนือ": "เหนือ", "ทางเหนือ": "เหนือ",
    "ภาคใต้": "ใต้", "โซนใต้": "ใต้", "แถบใต้": "ใต้", "ทางใต้": "ใต้",
    "ภาคกลาง": "กลาง", "โซนกลาง": "กลาง", "แถบกลาง": "กลาง",
    "ภาคตะวันออก": "ออก", "ภาคออก": "ออก", "โซนออก": "ออก", "อีสเทิร์น": "ออก", "อีสเทอร์น": "ออก",
    "ภาคอีสาน": "อีสาน", "โซนอีสาน": "อีสาน", "ภาคตะวันออกเฉียงเหนือ": "อีสาน", "ทางอีสาน": "อีสาน",
    "ภาคตะวันตก": "ตะวันตก", "โซนตะวันตก": "ตะวันตก",
}

# รายชื่อจังหวัดทั้ง 77 จังหวัด (ใช้ชื่อมาตรฐานหลัง alias แล้ว)
# ใช้ตรวจสอบว่า query หรือ dataset มีชื่อจังหวัดไหมบ้าง
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

# global variables โหลดตอน startup ใช้ร่วมกันทุก request
# docs     = list ของ Document ทั้งหมด (main + detail chunks)
# retriever = FAISS retriever พร้อมค้นหา
docs = None
retriever = None


# ================================================================
# 4) REQUEST MODEL — validate body ที่รับจาก frontend/Node.js
# ================================================================

class HistoryMessage(BaseModel):
    """1 ข้อความใน conversation history"""
    role: str     # "user" หรือ "assistant"
    content: str  # เนื้อหาข้อความ


class QuestionRequest(BaseModel):
    """body ของ POST /ask-pha"""
    question: str                              # คำถามปัจจุบัน
    history: Optional[List[HistoryMessage]] = []  # ประวัติการสนทนาก่อนหน้า


# ================================================================
# 5) INTENT DETECTION — ตัดสินใจว่าต้องดึง context งานไหม
#
# intent = "search"  → ดึง docs จาก FAISS แล้วสร้าง context
# intent = "general" → ไม่ดึง docs ให้ LLM ตอบจาก system prompt เอง
# ================================================================

# คีย์เวิร์ดที่บ่งบอกว่าผู้ใช้ต้องการค้นหางานอาสา
SEARCH_KEYWORDS = [
    "หางานอาสา", "อยากทำจิตอาสา", "มีงานอาสาไหม", "แนะนำงานอาสา",
    "กิจกรรมอาสา", "สมัครอาสา", "อยากช่วยสังคม",
    # เพิ่มคำพูดทั่วไปที่คนใช้จริง
    "หางาน", "อยากหางาน", "มีงานไหม", "งานอาสา",
    "อยากทำอาสา", "อยากอาสา", "อาสาสมัคร",
]

_SEARCH_EMBEDDINGS = None  # lazy-load ตอนใช้ครั้งแรก


def get_search_embeddings():
    """embed SEARCH_KEYWORDS ครั้งเดียวแล้ว cache ไว้ใน global variable"""
    global _SEARCH_EMBEDDINGS
    if _SEARCH_EMBEDDINGS is None:
        _SEARCH_EMBEDDINGS = np.vstack([embed_text(t) for t in SEARCH_KEYWORDS])
    return _SEARCH_EMBEDDINGS


# คีย์เวิร์ดที่บ่งบอกว่าผู้ใช้ถามเรื่องทั่วไป ไม่ใช่ค้นหางาน
GENERAL_KEYWORDS = [
    "สวัสดี", "คุณคือใคร", "จิตอาสาคืออะไร", "เริ่มทำอาสายังไง",
    "ประโยชน์ของจิตอาสา", "ต้องเตรียมอะไร",
]

_GENERAL_EMBEDDINGS = None


def detect_intent(question: str, history: list) -> str:
    """
    ตัดสินใจว่าคำถามนี้ต้องการค้นหางานหรือแค่ถามทั่วไป

    ลำดับการตรวจ (priority สูง → ต่ำ):
    1. มีชื่อจังหวัดใน query? → search (ชัดเจนที่สุด)
    2. มีชื่อภาค/โซน?        → search
    3. มี search keyword?     → search (ตรวจก่อน general เพื่อกัน false positive)
    4. มี general keyword?    → general
    5. ดู history ล่าสุด      → ถ้าเคยถามเรื่องงาน = search (ตอบต่อบทสนทนา)
    6. default                → general (ให้ LLM จัดการเอง)

    หมายเหตุ: ต้องตรวจ search ก่อน general เพราะ
    "อยากหางานแถวโคราช" มีคำว่า "อยาก" ซึ่งอาจ match general ก่อนถ้าไม่ระวัง
    """
    q = question.lower().strip()

    # ── Priority 1 & 2: จังหวัด/ภาค ─────────────────────────
    # ตรวจก่อนเพราะชัดเจนที่สุด ไม่มีทาง false positive
    if detect_province_in_query(question):
        return "search"
    if detect_region_in_query(question):
        return "search"

    # ── Priority 3: search keywords ──────────────────────────
    # ตรวจก่อน general เพื่อกัน "อยากหางาน..." ถูก detect เป็น general
    for kw in SEARCH_KEYWORDS:
        if kw in q:
            return "search"

    # ── Priority 4: general keywords ─────────────────────────
    for kw in GENERAL_KEYWORDS:
        if kw in q:
            return "general"

    # ── Priority 5: history context ──────────────────────────
    # รวม 8 เทิร์นล่าสุดของ user — ถ้าเคยถามเรื่องงาน การตอบต่อก็น่าจะเกี่ยวด้วย
    if history:
        recent = " ".join([h.content for h in history[-8:]
                           if h.role == "user"]).lower()
        for kw in SEARCH_KEYWORDS + ALL_PROVINCES:
            if kw in recent:
                return "search"

    # ── Priority 6: default ───────────────────────────────────
    return "general"


# ================================================================
# 6) HELPERS — ฟังก์ชันช่วยเหลือต่างๆ
# ================================================================

def extract_provinces(text: str) -> list:
    """
    หาจังหวัดทั้งหมดที่ปรากฏใน text
    คืน list ของชื่อจังหวัดมาตรฐาน (หลัง alias แล้ว) ไม่ซ้ำ
    ใช้ normalize_text ก่อน เพื่อแปลง "โคราช" → "นครราชสีมา" ก่อน match
    """
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
    """
    ทำความสะอาด text สำหรับใช้เป็น search_text ใน vector store
    ขั้นตอน:
      1. lowercase + strip
      2. แทนที่ alias จังหวัด
      3. tokenize ด้วย pythainlp (newmm engine)
      4. ลบ stopwords (ยกเว้นคำอาสาสำคัญที่ถูก remove ออกก่อนแล้ว)
      5. join กลับเป็น string คั่นด้วยช่องว่าง
    """
    if pd.isna(text) or text is None:
        return ""
    text = str(text).strip().lower()
    for k, v in PROVINCE_ALIAS.items():
        text = text.replace(k.lower(), v.lower())
    # newmm = เร็วและแม่นยำสำหรับภาษาไทยทั่วไป
    tokens = word_tokenize(text, engine="newmm")
    tokens = [t for t in tokens if t.strip() and t not in stopwords]
    return " ".join(tokens)


def normalize_text(text):
    """
    Normalize เบาๆ — strip + แทน alias จังหวัดแบบ case-insensitive
    ใช้ re.sub แทน str.replace เพื่อกัน case ที่ตัวพิมพ์ใหญ่/ผสม
    เช่น "โคราช" / "KORAT" / "โคราช" → "นครราชสีมา" ทุกกรณี

    ต่างจาก clean() ตรงที่ ไม่ tokenize และไม่ลบ stopwords
    ใช้กับ field ที่ต้องการเก็บ original text (title, location, ...)
    """
    if pd.isna(text) or text is None:
        return ""
    text = str(text).strip()
    for k, v in PROVINCE_ALIAS.items():
        text = re.sub(re.escape(k), v, text, flags=re.IGNORECASE)
    return text


def load_data(path):
    """โหลด dataset — รองรับทั้ง .json และ .csv"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"ไม่พบไฟล์ {path}")
    return pd.read_json(path) if path.endswith(".json") else pd.read_csv(path)


# ── วันที่: parse + กรองหมดอายุ ──────────────────────────────

# mapping ชื่อเดือนภาษาไทย (ทั้งแบบเต็มและแบบย่อ) → เลขเดือน
# ใช้ใน parse_event_end_date() เพื่อแปลง string วันที่ → date object
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
    Parse วันที่สิ้นสุดของงานจาก string ภาษาไทย
    รองรับหลายรูปแบบ เช่น:
      "06:00 เสาร์ 13 มิ.ย. 2569 - 14:00 เสาร์ 13 มิ.ย. 2569"
      "08:30-12:30 น. อาทิตย์ 24 เม.ย. 2569"

    หมายเหตุ: ปีใน dataset เป็น พ.ศ. → แปลง -543 → ค.ศ. ก่อนสร้าง date object
    คืน date object ของวันสุดท้าย (max) หรือ None ถ้า parse ไม่ได้
    """
    if not date_str or date_str == "ไม่ระบุ":
        return None
    try:
        # regex หาทุก pattern ที่เป็น "วัน เดือนภาษาไทย ปีพ.ศ." ใน string
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
            year_ce = year_be - 543  # พ.ศ. 2569 = ค.ศ. 2026
            dates.append(date(year_ce, month, day))
        # คืน max เพราะต้องการวันสุดท้ายของงาน
        # กัน filter งานหลายวันออกก่อนวันสุดท้ายจริงๆ
        return max(dates) if dates else None
    except Exception:
        return None


def is_expired(date_str: str, today: date | None = None) -> bool:
    """
    คืน True ถ้างานหมดแล้ว (วันสุดท้ายผ่านวันนี้ไปแล้ว)
    ถ้า parse วันที่ไม่ได้ → คืน False (เก็บไว้ก่อน ดีกว่าลบทิ้งผิด)
    """
    if today is None:
        today = datetime.now().date()
    end = parse_event_end_date(date_str)
    if end is None:
        return False
    return end < today  # วันสิ้นสุดอยู่ก่อนวันนี้ = หมดแล้ว


# ── Preprocess DataFrame ──────────────────────────────────────

def preprocess(df):
    """
    เตรียม DataFrame ก่อนสร้าง vector store
    ขั้นตอน:
      1. fillna("ไม่ระบุ") กัน NaN crash
      2. rename columns → ชื่อ alias สั้นกว่า
      3. แปลง cost เป็น binary (ฟรี/มีค่าใช้จ่าย)
      4. extract จังหวัดจากทุก field รวมกัน → เก็บใน "provinces"
      5. fix_location: ถ้า location = "ไม่ระบุ" แต่มีจังหวัด → ใช้จังหวัดแทน
      6. สร้าง _clean version ของทุก field ด้วย clean()
      7. รวมทุก field (raw + clean) เป็น "search_text" → ใช้เป็น embedding input
    """
    df = df.fillna("ไม่ระบุ").copy()

    # rename columns จาก JSON ภาษาไทย → ชื่อ alias สั้น
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

    # แปลง cost: False/false/0/"ไม่ระบุ" → ฟรี, อื่นๆ → มีค่าใช้จ่าย
    if "มีค่าใช้จ่าย" in df.columns:
        df["cost"] = df["มีค่าใช้จ่าย"].apply(
            lambda x: "ไม่เสียค่าใช้จ่าย"
            if x is False or str(x).lower() in ["false", "0", "ไม่ระบุ"]
            else "มีค่าใช้จ่าย"
        )
    else:
        df["cost"] = "ไม่ระบุ"

    def get_provinces(row):
        # รวม 3 field ที่มักมีชื่อจังหวัด แล้วสกัดจังหวัดออกมา
        # ตัด รายละเอียด ที่ 1000 ตัวกัน performance
        combined = " ".join([
            str(row.get("ชื่อกิจกรรม", "")),
            str(row.get("สถานที่", "")),
            str(row.get("รายละเอียด", ""))[:1000],
        ])
        return " ".join(extract_provinces(combined))

    df["provinces"] = df.apply(get_provinces, axis=1)

    def fix_location(row):
        # บางแถว location = "ไม่ระบุ" แต่มีจังหวัดในชื่องานหรือรายละเอียด
        # ให้ใช้จังหวัดที่สกัดได้แทน เพื่อให้ค้นหาตามพื้นที่ได้
        if row["location"] == "ไม่ระบุ" and row["provinces"]:
            return row["provinces"]
        return row["location"]

    df["location"] = df.apply(fix_location, axis=1)

    # สร้าง _clean version ของทุก field ที่ต้องการ tokenize
    for c in ["title", "org", "location", "date", "cost"]:
        df[f"{c}_clean"] = df[c].apply(clean)

    df["provinces_clean"] = df["provinces"].apply(clean)

    def combine_search(r):
        """
        สร้าง search_text ที่จะถูก embed และเก็บใน FAISS
        รวมทั้ง raw text และ tokenized text เพื่อให้:
          - raw  → embedding model จับ semantic ได้ดี (ประโยคสมบูรณ์)
          - clean → keyword match ได้แม่นยำขึ้น
        """
        detail_raw = normalize_text(str(r.get("detail", "")))[:400]
        detail_clean = clean(str(r.get("detail", "")))[:200]
        return " ".join([
            # raw fields (context ครบ)
            str(r.get("title", "")),
            str(r.get("org", "")),
            str(r.get("location", "")),
            str(r.get("provinces", "")),
            # cleaned fields (tokenized + ลบ stopwords)
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


# ── Build FAISS Vector Store ──────────────────────────────────

def build_vector(df):
    """
    สร้าง FAISS vector store จาก DataFrame ที่ผ่าน preprocess แล้ว

    สร้าง Document 2 แบบต่อ 1 แถวข้อมูล:
      1. "main" doc  — search_text ทั้งหมด + metadata ครบ
                       ใช้แสดงผลตอน build_context
      2. "detail" docs — ตัดรายละเอียดยาวๆ เป็น chunks
                         ช่วยให้ค้นหา keyword ในรายละเอียดได้

    FAISS retriever ใช้ MMR (Maximal Marginal Relevance):
      - fetch_k=100  : ดึงมา 100 docs ก่อน
      - k=15         : เลือก 15 ที่ diverse ที่สุด
      - lambda_mult=0.5 : balance relevance (1.0) กับ diversity (0.0)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # ลำดับ separator: ลองตัดด้วย \n\n ก่อน ถ้าไม่ได้ลอง \n ต่อไปเรื่อยๆ
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )

    all_docs = []

    for _, row in df.iterrows():
        # metadata จะติดไปกับทุก Document
        # ใช้ตอน build_context และ filter_docs
        metadata = {
            "title":     row.get("title", "ไม่ระบุ"),
            "org":       row.get("org", "ไม่ระบุ"),
            "location":  row.get("location", "ไม่ระบุ"),
            "date":      row.get("date", "ไม่ระบุ"),
            "cost":      row.get("cost", "ไม่ระบุ"),
            "provinces": row.get("provinces", ""),
            "url":       row.get("url", "ไม่ระบุ"),
            "doc_type":  "main",  # บอกว่าเป็น main doc (ใช้แยกจาก detail chunks)
        }

        # Main doc — 1 doc ต่อ 1 งาน
        all_docs.append(Document(
            page_content=row["search_text"],
            metadata=metadata,
        ))

        # Detail chunks — แตกรายละเอียดเป็นหลาย docs ถ้ายาวพอ
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

    db = FAISS.from_documents(all_docs, _embedding_model)
    local_retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 15, "fetch_k": 100, "lambda_mult": 0.5}
    )
    return all_docs, local_retriever


# ── Query Detection Helpers ───────────────────────────────────

def detect_region_in_query(q: str):
    """
    ตรวจว่า query ระบุภาค/โซนไหม
    คืน list ของจังหวัดในภาคนั้น หรือ [] ถ้าไม่พบ
    ตรวจ alias ก่อน (เช่น "ภาคเหนือ") แล้วค่อยตรวจชื่อตรง (เช่น "เหนือ")
    """
    q_lower = q.lower().strip()
    for alias, region_key in REGION_ALIAS.items():
        if alias in q_lower:
            return REGION_MAP.get(region_key, [])
    for region_key, provinces in REGION_MAP.items():
        if region_key in q_lower:
            return provinces
    return []


def detect_province_in_query(q: str):
    """
    ตรวจว่า query ระบุจังหวัดไหม
    คืน ชื่อจังหวัดมาตรฐาน (string) หรือ None
    normalize ก่อนตรวจ เพื่อจับ alias เช่น "กทม" → "กรุงเทพ"
    """
    q_norm = normalize_text(q)
    for prov in ALL_PROVINCES:
        if prov in q_norm:
            return PROVINCE_ALIAS.get(prov, prov)
    return None


def enhance_query(q):
    """
    เพิ่ม synonym เข้า query ก่อนส่ง FAISS
    เพื่อให้ match กับ text ใน dataset ที่อาจใช้คำต่างกัน
    เช่น "ฟรี" → เพิ่ม "ไม่เสียค่าใช้จ่าย"

    ส่งทั้ง raw normalized query + tokenized text
    เพราะ embedding model เข้าใจภาษาธรรมชาติได้ดีกว่า tokenized เพียงอย่างเดียว
    """
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

    tokens = word_tokenize(q_norm, engine="newmm")
    tokens = [t for t in tokens if t.strip() and t not in stopwords]
    tokenized = " ".join(tokens)

    # รวม raw + tokenized → semantic + keyword match
    return f"{q_norm} {tokenized}".strip()


# ── Skill Detection ───────────────────────────────────────────

# mapping ทักษะที่ผู้ใช้บอก → keywords ที่ใช้กรองงานใน dataset
SKILL_KEYWORDS = {
    "ก่อสร้าง": ["ก่อสร้าง", "ซ่อม", "ช่าง", "สร้างบ้าน"],
    "สอน":      ["สอน", "ติว", "ครู", "การศึกษา"],
    "ออกแบบ":   ["ออกแบบ", "กราฟิก", "วาดรูป"],
    "แพทย์":    ["แพทย์", "พยาบาล", "สุขภาพ"],
    "IT":        ["โปรแกรม", "คอมพิวเตอร์", "เทคโนโลยี"],
}

# pre-embed ทุก skill keyword ตอน startup
# เพื่อไม่ต้อง embed ซ้ำทุก request
_SKILL_EMBEDDINGS = {
    skill: np.vstack([embed_text(k) for k in kws])
    for skill, kws in SKILL_KEYWORDS.items()
}


def detect_skill_keywords(q: str, threshold=0.55) -> list:
    """
    ตรวจว่า query มี skill ใดบ้าง โดยใช้ cosine similarity
    threshold=0.55 → ต้องคล้ายพอสมควร ไม่ใช่แค่มีคำเหมือนกัน
    คืน list ของ skill names ที่ match
    """
    q_vec = embed_text(q)
    result = []
    for skill, vecs in _SKILL_EMBEDDINGS.items():
        sims = cosine_similarity(q_vec, vecs)[0]
        if sims.max() > threshold:
            result.append(skill)
    return result


# ── Filter, Deduplicate, Build Context ───────────────────────

def filter_docs(found_docs, q, locked_province: str = None):
    """
    กรอง docs ตาม preference ของผู้ใช้และกรองงานหมดอายุออก

    MODE 1 — มีจังหวัด/ภาค (want_province หรือ want_region_provinces):
      → scan global docs ทั้งหมด กรองตาม province
      → เร็วและแม่นกว่า vector search สำหรับ province filter
      → ถ้าหาไม่เจอ คืน [] เพื่อให้ LLM บอก user ว่าหาไม่เจอ

    MODE 2 — มี hard filter (ออนไลน์/ไม่ออนไลน์/ฟรี):
      → scan global docs เพราะ vector search อาจไม่คืน docs ที่ผ่าน filter

    DEFAULT — ไม่มีทั้งสองอย่าง:
      → filter จาก found_docs ที่ FAISS คืนมา
      → fallback: ถ้าผล < 2 → คืน found_docs เดิม กัน 0 result
    """
    q_norm = normalize_text(q).lower()

    # ── ตรวจ preference จาก query ────────────────────────────
    want_free = "ฟรี" in q_norm or "ไม่เสียค่า" in q_norm or "ไม่มีค่า" in q_norm

    _online_want_kws = ["ออนไลน์", "ทำที่บ้าน", "work from home", "remote", "ทำออนไลน์", "อยู่บ้าน"]
    want_online = any(k in q_norm for k in _online_want_kws)

    _not_online_kws = [
        "ไม่ออนไลน์", "ออนไซต์", "ไม่ทำที่บ้าน", "ไม่เอาที่บ้าน",
        "ออกไปข้างนอก", "ออกนอก", "ไปทำ", "ไปที่", "ออกไปทำ",
        "นอกบ้าน", "ไม่ work from home", "ไม่ remote",
        "อยากออกไป", "ออกไปเจอคน", "เจอคน", "เจอผู้คน",
        "ไม่อยู่บ้าน", "ไม่เอาออนไลน์"
    ]
    want_not_online = any(k in q_norm for k in _not_online_kws)

    # locked_province มาจาก history (เคยระบุจังหวัดในเทิร์นก่อน)
    want_province = locked_province or detect_province_in_query(q)
    want_region_provinces = [] if want_province else detect_region_in_query(q)
    skill_kws = detect_skill_keywords(q)

    # keywords บ่งชี้งานออนไลน์ในเนื้อหางาน
    online_keywords = ["ออนไลน์", "online", "remote", "ทำที่บ้าน", "work from home"]
    today = datetime.now().date()

    # ── MODE 1: มีจังหวัดหรือภาค ─────────────────────────────
    if want_province or want_region_provinces:
        global docs
        results = []
        # สร้าง list จังหวัดที่ต้องการ (lowercase สำหรับ string comparison)
        filter_provinces = [want_province.lower()] if want_province else [
            p.lower() for p in want_region_provinces]

        for d in docs:
            # เฉพาะ main doc — กัน detail chunk duplicate
            if d.metadata.get("doc_type") != "main":
                continue
            md = d.metadata

            # กรองงานหมดอายุออก
            if is_expired(md.get("date", ""), today):
                continue

            # normalize_text ก่อน lowercase → "โคราช" → "นครราชสีมา" ก่อนเทียบ
            merged = normalize_text(" ".join([
                str(md.get("provinces", "")),
                str(md.get("location", "")),
                str(md.get("title", "")),
                str(md.get("org", "")),
                (d.page_content or ""),
            ])).lower()

            if not any(p in merged for p in filter_provinces):
                continue  # ไม่มีจังหวัดที่ต้องการ

            is_free = "ไม่เสียค่าใช้จ่าย" in str(md.get("cost", "")).lower()
            is_online = any(k in merged for k in online_keywords)

            # apply preference filter เพิ่มเติม
            if want_free and not is_free:
                continue
            if want_online and not is_online:
                continue
            if want_not_online and is_online:
                continue

            results.append(d)

        label = want_province if want_province else f"ภาค ({len(want_region_provinces)} จังหวัด)"
        print(f"พบงานใน {label} (ไม่หมดอายุ) = {len(results)} งาน")

        if not results:
            # คืน [] เพื่อให้ LLM รู้ว่าหาไม่เจอ แล้วบอก user พร้อมเสนอทางเลือก
            print(f"[FALLBACK] province mode ไม่เจองาน → คืน empty list")
        return results

    # ── MODE 2: มี hard filter (ออนไลน์/ฟรี) ────────────────
    # scan global docs แทน found_docs เพราะ vector search อาจไม่คืน docs ที่ผ่าน filter
    search_pool = docs if (want_not_online or want_online or want_free) else found_docs

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
        # ถ้ามี skill filter → ต้อง match อย่างน้อย 1 keyword
        if skill_kws:
            if not any(k in merged for k in skill_kws):
                continue
        result.append(d)

    has_hard = want_free or want_online or want_not_online
    # fallback: ถ้าไม่มี hard filter และผลน้อยกว่า 2 → คืน vector result เดิม
    if len(result) < 2 and not has_hard:
        return [d for d in found_docs if not is_expired(d.metadata.get("date", ""), today)]
    return result


def deduplicate_docs(found_docs):
    """
    ลบ docs ซ้ำที่ URL หรือ title เดียวกัน
    เรียง main ก่อน detail เสมอ เพื่อกัน main doc ถูกตัดทิ้ง
    เพราะ FAISS อาจคืน detail chunk ก่อน main doc ทำให้
    deduplicate ตัด main ทิ้ง → build_context หา title ไม่เจอ → แสดงงานผิด
    """
    # main doc ก่อน detail เสมอ (sort key: 0 = main, 1 = detail)
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
    สร้าง context string จาก docs ที่กรองแล้ว สำหรับส่งให้ LLM
    รูปแบบ: [งานที่ N] ชื่อ/องค์กร/สถานที่/วันที่/ค่าใช้จ่าย/ลิงก์

    ใช้เฉพาะ main docs (มี title) max 5 งาน
    deduplicate ตาม url อีกชั้น กัน url ซ้ำที่ deduplicate_docs พลาด
    """
    lines = []
    count = 0
    seen_urls = set()

    for d in found_docs:
        if count >= max_items:
            break
        md = d.metadata

        # ใช้เฉพาะ main doc — กัน detail chunk หลุดมาแสดง
        if md.get("doc_type") != "main":
            continue

        title = md.get("title", "")
        if not title or title == "ไม่ระบุ":
            continue

        # กัน url ซ้ำชั้นสุดท้าย
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


# ================================================================
# 7) SYSTEM PROMPT — บุคลิกและกฎของ "ภา"
#
# prompt นี้ถูกส่งเป็น system message ทุก request
# กำหนดโทน บุคลิก สิ่งที่ทำได้/ไม่ได้ และรูปแบบการตอบ
# ================================================================
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
- ถ้าผู้ใช้บอกว่า "ไม่รู้จะเลือกอันไหน" หรือ "ชอบหลายอัน" → ถามว่า "อยากทำวันไหน?" หรือ "ชอบทำกับกลุ่มไหนมากกว่า?"
- ถ้าผู้ใช้ถามว่า "ทำไมถึงแนะนำอันนี้?" → อธิบายสั้นๆ ว่าตรงกับที่ถามยังไง
- ถ้าผู้ใช้บอกว่า "ไม่ชอบอันนี้" → ถามว่า "ไม่ชอบตรงไหนคะ?" แล้วหาให้ใหม่

=== เมื่อแสดงงานแล้วไม่มีผลลัพธ์ ===
อย่าแค่บอกว่า "หาไม่เจอ" ให้:
1. บอกว่าหาไม่เจอ + สาเหตุ (เช่น "ตอนนี้ยังไม่มีงานก่อสร้างในเชียงใหม่")
2. เสนอทางเลือก (เช่น "ลองดูงานก่อสร้างจังหวัดอื่นๆ ไหม?")
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


# ================================================================
# 8) BUILD GROQ MESSAGES — สร้าง messages array ส่งให้ Groq API
# ================================================================

def build_groq_messages(question: str, context: str, history: list) -> list:
    """
    สร้าง messages array (OpenAI format) สำหรับส่งให้ Groq API

    โครงสร้าง:
      [system]    SYSTEM_PERSONA
      [user]      history[0].content   ← เทิร์นเก่าสุด
      [assistant] history[1].content
      ...
      [user]      คำถามปัจจุบัน + context (ถ้า intent=search)

    ถ้า intent=search → แนบ context งานก่อนส่ง พร้อม instruction "ห้ามแต่งข้อมูล"
    ถ้า intent=general → ส่งแค่คำถามเปล่า ให้ LLM ตอบจาก system prompt
    """
    messages = [{"role": "system", "content": SYSTEM_PERSONA}]

    # ใส่ history ทุกเทิร์น — LLM จำบริบทได้ตลอด conversation
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


# ================================================================
# 9) GROQ STREAM GENERATOR — stream คำตอบจาก Groq API
# ================================================================

async def groq_stream_generator(question: str, context: str, history: list):
    """
    async generator ที่ yield text ทีละ chunk จาก Groq API

    Groq ส่งกลับเป็น SSE (Server-Sent Events) format:
      data: {"choices":[{"delta":{"content":"ข้"}}]}
      data: {"choices":[{"delta":{"content":"้อ"}}]}
      ...
      data: [DONE]

    Key Rotation:
      ถ้าเจอ 429 (rate limit) หรือ 401 (key ไม่ valid)
      → สลับไปใช้ key ถัดไปแล้ว retry อัตโนมัติ
      → ลองได้สูงสุด len(GROQ_API_KEYS) ครั้ง
      → ถ้าหมดทุก key → yield ข้อความแจ้ง user (ไม่ raise error)
    """
    messages = build_groq_messages(question, context, history)
    body = {
        "model": GROQ_MODEL,
        "messages": messages,
        "max_tokens": 600,   # จำกัดความยาวคำตอบ กัน token หมดเร็ว
        "temperature": 0.5,  # 0=deterministic, 1=creative — 0.5 สมดุล
        "stream": True,      # บอก Groq ให้ส่งแบบ streaming
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
            # AsyncClient ใช้ใน async context, timeout=60s
            async with httpx.AsyncClient(timeout=GROQ_TIMEOUT) as client:
                # client.stream() → เปิด connection แบบ streaming
                async with client.stream(
                    "POST", "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers, json=body
                ) as r:
                    if r.status_code in (429, 401):
                        # 429 = rate limit, 401 = key ไม่ valid → สลับ key
                        print(f"[GROQ] key #{_current_key_index + 1} ถูก block ({r.status_code}) → สลับ key")
                        get_next_groq_key()
                        tried += 1
                        continue
                    r.raise_for_status()
                    # อ่านทีละ line (แต่ละ line = 1 SSE event)
                    async for line in r.aiter_lines():
                        if line.startswith("data:"):
                            payload = line[5:].strip()  # ตัด "data:" ออก
                            if not payload or payload == "[DONE]":
                                continue  # [DONE] = stream จบแล้ว
                            try:
                                chunk = json.loads(payload)
                                text = chunk["choices"][0]["delta"].get("content", "")
                                if text:
                                    yield text  # ส่งออกทีละ chunk ทันที
                            except Exception:
                                continue  # JSON parse error → ข้ามไป
                    return  # stream จบสมบูรณ์

        except httpx.HTTPStatusError as e:
            if e.response.status_code in (429, 401):
                print(f"[GROQ] key #{_current_key_index + 1} error {e.response.status_code} → สลับ key")
                get_next_groq_key()
                tried += 1
                last_error = e
            else:
                raise  # error อื่น (500 ฯลฯ) → ปล่อยขึ้นไป handle ข้างบน

    # ทุก key หมด → yield ข้อความแทน raise เพื่อให้ user เห็นคำตอบแทน crash
    if last_error and hasattr(last_error, 'response') and last_error.response.status_code == 429:
        yield "ขอโทษนะคะ ตอนนี้ภาโดนจำกัดการใช้งานชั่วคราว 🙏 ลองใหม่อีกสักครู่ได้เลยค่ะ"
    else:
        yield "เชื่อมต่อไม่ได้ตอนนี้ ลองใหม่อีกทีนะคะ"


# ================================================================
# 10) RAG HELPERS — ฟังก์ชันช่วย ask_rag()
# ================================================================

def extract_province_from_history(history: list) -> str | None:
    """
    ย้อนดู history เพื่อหาจังหวัดล่าสุดที่ผู้ใช้เคยระบุ
    ใช้ "lock" จังหวัดข้ามเทิร์น เช่น:
      เทิร์น 1: "มีงานเชียงใหม่ไหม"
      เทิร์น 2: "มีงานสอนเด็กไหม" ← ยังคง filter เชียงใหม่อยู่
    reversed() = ย้อนจากเทิร์นล่าสุด เพื่อได้จังหวัดที่ recent ที่สุด
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
    RAG Pipeline หลัก (Retrieval-Augmented Generation)

    ขั้นตอน:
      1. detect_intent() → ถ้า general คืน context ว่าง ไม่ต้อง retrieve
      2. detect_province จาก query + history (province locking)
      3. Province mode: filter จาก global docs ตรงๆ (เร็วและแม่นกว่า vector)
      4. Vector mode: FAISS MMR search → filter_docs() → deduplicate
      5. build_context() → สร้าง string ส่งให้ LLM

    คืน (context_str, found_docs)
    """
    global docs, retriever

    intent = detect_intent(question, history or [])
    print(f"intent = {intent}")

    # general intent → ไม่ต้องดึง context เลย
    if intent == "general":
        return "", []

    # ตรวจจังหวัดจากคำถามปัจจุบัน ถ้าไม่มีให้ดูจาก history
    current_province = detect_province_in_query(question)
    locked_province = current_province or extract_province_from_history(history or [])
    print(f"province: current={current_province}, locked={locked_province}")

    if locked_province:
        # Province mode: scan global docs โดยตรง ไม่ผ่าน FAISS
        print(f"Province mode: {locked_province}")
        found = filter_docs([], question, locked_province=locked_province)
    else:
        # Vector mode: รวม history เข้า query เพื่อให้ embedding จับบริบทได้ดีขึ้น
        context_query = question
        if history:
            # ใช้ 6 เทิร์นล่าสุดของ user
            prev_user = " ".join([h.content for h in history[-6:] if h.role == "user"])
            context_query = prev_user + " " + question

        query = enhance_query(context_query)  # normalize + เพิ่ม synonym
        found = retriever.invoke(query)        # FAISS MMR search
        print(f"question={question}, enhanced={query}, before filter={len(found)}")
        found = filter_docs(found, context_query)

    found = deduplicate_docs(found)
    print(f"after dedup = {len(found)}")
    context = build_context(found)
    return context, found


# ================================================================
# 11) STARTUP — โหลดข้อมูลครั้งเดียวตอนเริ่ม server
# ================================================================

@app.on_event("startup")
def startup_event():
    """
    โหลด dataset + สร้าง FAISS vector ตอน server start
    เก็บไว้ใน global variable docs, retriever
    ทุก request ใช้ตัวเดิมร่วมกัน ไม่โหลดซ้ำทุกครั้ง
    """
    global docs, retriever
    df = preprocess(load_data(DATASET_PATH))
    docs, retriever = build_vector(df)
    print("โหลดข้อมูลและสร้าง vector database เรียบร้อยแล้ว")


# ================================================================
# 12) ROUTES — API endpoints
# ================================================================

@app.get("/")
def root():
    """Health check endpoint — ตรวจว่า server ยังทำงานอยู่"""
    return {"message": "JitArsa backend is running"}


@app.post("/reload-data")
def reload_data():
    """
    เรียกจาก update_database.py หลัง scrape ข้อมูลใหม่เสร็จ
    rebuild FAISS vector ใหม่ทั้งหมดโดยไม่ต้อง restart server
    global variable ถูก replace → request ถัดไปใช้ข้อมูลใหม่ทันที
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
    """
    Endpoint หลัก — รับคำถามจาก frontend แล้ว stream คำตอบกลับ

    Flow:
      1. รับ question + history
      2. ask_rag() → ได้ context จาก dataset
      3. groq_stream_generator() → stream คำตอบจาก Groq
      4. StreamingResponse → ส่ง chunk กลับ frontend แบบ real-time

    StreamingResponse: FastAPI flush ทุก chunk ทันที
    ไม่รอให้คำตอบเสร็จก่อน → user เห็นตัวอักษรทยอยออกมา
    """
    try:
        history = data.history or []

        # DEBUG log: ตรวจว่า history ที่รับมาครบไหม
        print(f"[DEBUG] question={data.question!r}")
        print(f"[DEBUG] history len={len(history)}")
        for i, h in enumerate(history):
            print(f"[DEBUG]   history[{i}] role={h.role!r} content={h.content[:60]!r}")

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