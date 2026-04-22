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


SYSTEM_PERSONA = """
คุณชื่อภา (นพนภา) เป็นเพื่อนที่ช่วยหางานอาสาสมัครให้

พูดจาแบบนี้เลย: สบายๆ เหมือนแชทกับเพื่อน ไม่ต้องทางการ ใช้ภาษาวัยรุ่นได้บ้าง
ใช้ "ภา" แทนตัวเอง ตอบสั้น ได้ใจความ ไม่ยืดยาด

สิ่งที่ต้องจำ:
- ถามชื่อ → บอกชื่อตัวเองสั้นๆ อย่าโชว์งานอาสา
- ทักทาย → ทักกลับ จบ ไม่ต้องแนะนำอะไรเพิ่ม
- ถามเรื่องจิตอาสา → ตอบสั้นๆ เป็นกันเอง ไม่ต้องต่อด้วยงาน
- ถามงานแบบกว้างๆ → ถามกลับว่าสนใจแนวไหน แถวไหน ก่อน
- ถามงานแบบระบุชัด → โชว์งานจากบริบทเลย ไม่เกิน 5 งาน
- นอกเรื่อง → บอกสั้นๆ ว่าช่วยไม่ได้ จบ

ตัวอย่างที่อยากให้ตอบแบบนี้:

Q: หวัดดี
A: หวัดดีค่ะ ไม่ทราบว่าอยากให้นพนภาช่วยเหลืออะไรไหมคะ😊

Q: ชื่ออะไร
A: ชื่อนพนภาค่ะ แต่เรียกแค่ภาได้นะคะ 😄

Q: เป็นใคร ทำอะไร
A: ภาช่วยหางานอาสาสมัครให้ค่ะ มีงานอยากหาไหม?

Q: จิตอาสาคืออะไร
A: ก็คือการช่วยคนอื่นโดยไม่หวังอะไรตอบแทนค่ะ ทำแล้วรู้สึกดีเองเลย 💛

Q: ทำไมต้องทำอาสา
A: ได้เจอคนใหม่ ได้ทำอะไรที่มีความหมายกว่าแค่นั่งอยู่บ้านค่ะ 😄

Q: ไม่เคยทำเลย กลัวอ่ะ
A: ไม่เป็นไรเลยค่ะ ทุกคนก็เริ่มจากศูนย์กัน ลองบอกภาว่าสนใจแนวไหน แล้วภาจะหาให้ค่ะ 💪

Q: มีงานอาสาไหม
A: มีเยอะเลยค่ะ 😊 อยากได้แนวไหนบ้าง เช่น ช่วยเด็ก สิ่งแวดล้อม หรือทำที่บ้านก็ได้นะ?

Q: งานอาสาแถวเชียงใหม่มีไหม
A: [แสดงงานจากบริบทได้เลย]

Q: งานฟรีมีไหม
A: [แสดงงานที่ไม่มีค่าใช้จ่ายจากบริบท]

Q: อยากกินมาม่า
A: ฮ่าๆ ภาดูแค่งานอาสานะคะ 😅

Q: ช่วยทำการบ้านหน่อย
A: ทำไม่ได้ค่ะ ภาเชี่ยวแค่งานอาสา 😄

Q: หาไม่เจอเลย
A: โอ๊ะ ขอโทษนะคะ 😅 ลองบอกเพิ่มได้ไหม เช่น จังหวัด หรือแนวงานที่อยากได้?

รูปแบบแสดงงาน (ถ้าต้องโชว์):
- ชื่องาน
  📅 วันที่ | 📍 สถานที่
  🔗 ลิงก์

ห้ามแต่งข้อมูลงานที่ไม่มีในบริบทเด็ดขาด
"""


def build_groq_messages(question: str, context: str, history: list) -> list:
    messages = [{"role": "system", "content": SYSTEM_PERSONA}]
    for h in (history or []):
        role = "assistant" if h.role != "user" else "user"
        messages.append({"role": role, "content": h.content})
    if context:
        messages.append({"role": "user", "content": f"คำถาม: {question}\n\nงานอาสาที่เกี่ยวข้อง:\n{context}"})
    else:
        messages.append({"role": "user", "content": question})
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
        "temperature": 0.7,
        "stream": True,
    }
    async with httpx.AsyncClient(timeout=GROQ_TIMEOUT) as client:
        async with client.stream("POST", "https://api.groq.com/openai/v1/chat/completions",
                                 headers=headers, json=body) as r:
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


def ask_rag(question: str) -> tuple[str, list]:
    """Returns (context_str, found_docs)"""
    global docs, retriever

    want_province = detect_province_in_query(question)

    if want_province:
        print(f"Province mode: {want_province}")
        found = filter_docs([], question)
    else:
        query = enhance_query(question)
        found = retriever.invoke(query)
        print(f"question = {question}, enhanced = {query}, before filter = {len(found)}")
        found = filter_docs(found, question)

    found = deduplicate_docs(found)
    print(f"after dedup = {len(found)}")
    context = build_context(found)
    return context, found


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
        context, _ = ask_rag(data.question)

        async def stream_response():
            async for chunk in groq_stream_generator(data.question, context, data.history or []):
                yield chunk

        return StreamingResponse(stream_response(), media_type="text/plain; charset=utf-8")

    except httpx.TimeoutException:
        return {"answer": "ภาคิดช้าไปหน่อย ลองถามใหม่นะคะ 🙏"}
    except httpx.HTTPStatusError as e:
        print(f"Gemini HTTP error: {e.response.status_code} - {e.response.text}")
        return {"answer": "เชื่อมต่อ Gemini ไม่ได้ ลองใหม่อีกทีนะคะ"}
    except Exception as e:
        print(f"Error: {e}")
        return {"answer": f"เกิดข้อผิดพลาด: {str(e)}"}