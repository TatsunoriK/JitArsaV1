import httpx
import os
import requests
import pandas as pd
import json
import sys
import io
import logging
import httpx

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

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"
OLLAMA_TIMEOUT = 300

# Chunk config
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

# จังหวัด canonical ทั้งหมด
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
# 5) HELPERS
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

    # search_text สำหรับ doc หลัก (ไม่มี detail)
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

        # metadata fields รวมกัน
        all_docs.append(Document(
            page_content=row["search_text"],
            metadata=metadata,
        ))

        # chunk จาก field รายละเอียด
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


def filter_docs(found_docs, q):
    q_norm = normalize_text(q).lower()
    want_free = "ฟรี" in q_norm
    want_online = "ออนไลน์" in q_norm or "ทำที่บ้าน" in q_norm
    want_not_online = "ไม่ออนไลน์" in q_norm or "ออนไซต์" in q_norm
    want_province = detect_province_in_query(q)
    online_keywords = ["ออนไลน์", "online",
                       "remote", "ทำที่บ้าน", "work from home"]

    if want_province:
        global docs
        results = []
        for d in docs:
            # doc หลักสำหรับ province filter ป้องกัน duplicate จาก chunks
            if d.metadata.get("doc_type") != "main":
                continue
            md = d.metadata
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
        print(f"พบงานใน {want_province} = {len(results)} งาน")
        return results

    result = []
    for d in found_docs:
        md = d.metadata
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
        return found_docs
    return result


def deduplicate_docs(found_docs):
    """Deduplicate โดยใช้ url เป็น key — chunk หลายอันของงานเดียวกันจะเหลือแค่อันเดียว"""
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


def ask_llm(question: str, context: str) -> str:
    system_persona = """
คุณคือ "น้องภา" (Nop Napha) อาสาสมัครอัจฉริยะที่ร่าเริง แจ่มใส และใจดี 
หน้าที่ของคุณคือช่วยหาข้อมูลงานอาสาจากบริบทที่ให้ไว้ 

สไตล์การตอบกลับ:
- ใช้ภาษาที่เป็นกันเองเหมือนเพื่อน (เช่น ใช้คำว่า "จ้า", "น้า", "นะคะ", "ภาว่า...")
- หากคำถามเป็นการทักทาย (เช่น สวัสดี, ทำอะไรอยู่) ให้ตอบทักทายอย่างร่าเริงก่อนแล้วค่อยชวนเข้าเรื่องงานอาสา
- สามารถแสดงความเห็นใจหรือให้กำลังใจได้เล็กน้อย (เช่น "ว้าว งานนี้น่าสนุกมากเลยค่ะ", "สู้ๆ นะคะ ภาเป็นกำลังใจให้")
- หากไม่พบข้อมูลงานอาสาที่ตรงใจ ให้ตอบอย่างสุภาพและลองแนะนำงานที่ใกล้เคียง หรือบอกว่า "เสียดายจัง ตอนนี้ยังไม่มีงานแบบนั้นเลย แต่ลองดูงานพวกนี้แทนไหมคะ?"
- **ห้าม** ตอบสั้นห้วนเกินไป และ **ห้าม** หลุดจากบทบาทน้องภา
"""

    if not context:
        prompt = f"{system_persona}\n\nคำถาม: {question}\n(หมายเหตุ: คำถามนี้อาจเป็นการชวนคุยทั่วไป ให้คุณตอบในฐานะน้องภาอย่างเป็นธรรมชาติ)"
    else:
        prompt = f"{system_persona}\n\nคำถาม: {question}\nบริบทงานอาสา:\n{context}"

    return prompt


async def ollama_stream_generator(prompt):
    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
        async with client.stream("POST", OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": True
        }) as r:
            async for line in r.aiter_lines():
                if line:
                    chunk = json.loads(line)
                    content = chunk.get("response", "")
                    if content:
                        yield content


def ask_rag(question: str) -> str:
    global docs, retriever

    want_province = detect_province_in_query(question)

    if want_province:
        print(f"Province mode: {want_province}")
        found = filter_docs([], question)
    else:
        query = enhance_query(question)
        found = retriever.invoke(query)
        print(f"question = {question}")
        print(f"enhanced query = {query}")
        print(f"before filter = {len(found)}")
        found = filter_docs(found, question)

    found = deduplicate_docs(found)
    print(f"after dedup = {len(found)}")

    context = build_context(found)
    return ask_llm(question, context)


# ===============================
# 6) STARTUP
# ===============================
@app.on_event("startup")
def startup_event():
    global docs, retriever
    df = preprocess(load_data(DATASET_PATH))
    docs, retriever = build_vector(df)
    print("โหลดข้อมูลและสร้าง vector database เรียบร้อยแล้ว")


# ===============================
# 7) ROUTES
# ===============================
@app.get("/")
def root():
    return {"message": "JitArsa backend is running"}


@app.post("/ask-pha")
async def ask_api(data: QuestionRequest):

    try:
        prompt = ask_rag(data.question)
        if not prompt:
            return {"answer": "ภาหาข้อมูลที่เกี่ยวข้องไม่เจอเลย ลองถามแบบอื่นดูไหม?"}

        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            response = await client.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False
                }
            )

        response.raise_for_status()
        result = response.json()

        final_answer = result.get("response", "").strip()

        if not final_answer:
            print("Debug: Ollama returned empty response")
            return {"answer": "ขอโทษนะ ภามึนไปนิดนึง เลยยังไม่มีคำตอบให้เลย"}

        return {"answer": final_answer}

    except httpx.TimeoutException:
        return {"answer": "ภาใช้เวลาคิดนานเกินไป (Timeout) ลองถามสั้นลงหน่อยนะ"}
    except httpx.HTTPStatusError as e:
        print(
            f"Ollama HTTP error: {e.response.status_code} - {e.response.text}")
        return {"answer": f"Ollama ตอบกลับผิดปกติ (status {e.response.status_code})"}
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return {"answer": f"เกิดข้อผิดพลาดภายใน: {str(e)}"}
