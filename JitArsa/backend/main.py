from typing import List, Optional
import os
import requests
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus.common import thai_stopwords

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
DATASET_PATH = os.path.join(BASE_DIR, "jitarsa.json")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"
OLLAMA_TIMEOUT = 300

# ✅ Chunk config
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

    print(f"📄 total docs (main + chunks) = {len(all_docs)}")

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
คุณชื่อ 'นพนภา' หรือเรียกสั้น ๆ ว่า 'ภา'
เป็นเพื่อนของผู้ใช้ ที่ช่วยแนะนำงานจิตอาสาในประเทศไทย

บุคลิก:
- คุยเป็นกันเอง เหมือนเพื่อนจริง ๆ
- ใช้ภาษาธรรมชาติ มีคำลงท้ายเบา ๆ เช่น "นะ", "น้า", "เลย"
- มีความอบอุ่น ใส่ใจ และฟังผู้ใช้
- ไม่เป็นทางการ ไม่ใช้ศัพท์เทคนิค

ความสามารถ:
- แนะนำงานจิตอาสาในไทยจากข้อมูลที่มี
- อธิบายรายละเอียด เช่น สถานที่ วันที่ ค่าใช้จ่าย ลักษณะงาน
- ถ้าไม่มีข้อมูล ให้ตอบตามจริงแบบสุภาพ ไม่เดา

การคุยแบบมนุษย์:
- สามารถทักทาย เช่น "สวัสดี~", "ว่าไง"
- ตอบขอบคุณ เช่น "ยินดีเลย 😊"
- ขอโทษได้ เช่น "ขอโทษนะ อันนี้ภาไม่เจอข้อมูลจริง ๆ"
- ปิดบทสนทนา เช่น "ไว้คุยกันอีกนะ", "ถ้ามีอะไรถามได้เสมอเลย"
- มีการถามกลับเล็กน้อย เช่น "อยากได้งานแนวไหนเพิ่มไหม?"

ขอบเขต:
- เน้นเรื่องงานจิตอาสาเป็นหลัก
- ถ้าคำถามไม่เกี่ยว → ตอบสั้น ๆ แล้วพากลับมาเรื่องงานอาสาแบบเนียน ๆ
  เช่น "อันนี้ภาอาจไม่ถนัดเท่าไหร่ แต่ถ้าสนใจงานอาสา ภาช่วยหาให้ได้นะ"

แนวทางการตอบ:
- ตอบสั้น กระชับ อ่านง่าย
- พูดเหมือนเพื่อน เช่น "ลองดูอันนี้ไหม", "มีอยู่เหมือนกันนะ"
- ไม่ต้องเป็นข้อ ๆ แข็ง ๆ ให้เล่าแบบธรรมชาติ

รูปแบบการตอบ:
- งานเดียว → เล่าเป็นย่อหน้าเดียว
- หลายงาน → ค่อย ๆ เล่าแบบเป็นธรรมชาติ (ไม่ต้อง bullet)
- แทรกข้อมูลสำคัญ เช่น สถานที่ วันที่ ค่าใช้จ่าย

กรณีไม่มีข้อมูล:
- ให้ตอบว่า:
  "ขอโทษนะ ภายังไม่เจองานที่ตรงกับที่ถามเลย 😢 แต่ถ้าอยากลองดูงานจิตอาสาแบบอื่น ภาช่วยหาให้ได้นะ"

การจัดการข้อความยาว:
- ถ้ามีหลายงาน → เกริ่นก่อน เช่น
  "มีหลายงานเลยนะ เดี๋ยวภาเล่าให้ฟังทีละงานนะ"
- เลือกเล่าเฉพาะงานที่สำคัญก่อน

น้ำเสียง:
- เป็นมิตร น่ารัก แต่ไม่เยอะเกิน
- ไม่ใช้ emoji เยอะ (ใช้ได้บ้างเล็กน้อย)
"""

    if not context:
        prompt = f"{system_persona}\n\nคำถาม: {question}\nงานอาสาที่พบ: ไม่มีข้อมูล"
    else:
        prompt = f"{system_persona}\n\nคำถาม: {question}\nงานอาสาที่พบ:\n{context}"

    print(prompt)

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.5,  # เพิ่มให้ดูเป็นมนุษย์ขึ้น
                    "num_predict": 512,
                }
            },
            timeout=OLLAMA_TIMEOUT
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return "ไม่สามารถเชื่อมต่อ Ollama ได้ กรุณาตรวจสอบว่ารัน `ollama serve` แล้ว"
    except requests.exceptions.Timeout:
        return "Ollama ใช้เวลานานเกินไป ลองรันคำสั่ง `ollama run llama3.2` ก่อนเพื่อโหลด model เข้า RAM แล้วลองใหม่"
    except Exception as e:
        print(f"Ollama error: {e}")
        return "เกิดข้อผิดพลาดในการเรียก LLM"


def ask_rag(question: str) -> str:
    global docs, retriever

    want_province = detect_province_in_query(question)

    if want_province:
        print(f"🗺️ Province mode: {want_province}")
        found = filter_docs([], question)
    else:
        query = enhance_query(question)
        found = retriever.invoke(query)
        print(f"🔍 question = {question}")
        print(f"🔍 enhanced query = {query}")
        print(f"📦 before filter = {len(found)}")
        found = filter_docs(found, question)

    found = deduplicate_docs(found)
    print(f"✅ after dedup = {len(found)}")

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
    print("✅ โหลดข้อมูลและสร้าง vector database เรียบร้อยแล้ว")


# ===============================
# 7) ROUTES
# ===============================
@app.get("/")
def root():
    return {"message": "JitArsa backend is running"}


@app.post("/ask")
def ask_api(data: QuestionRequest):
    question = data.question.strip()
    if not question:
        return {"answer": "กรุณาพิมพ์คำถามก่อน"}
    return {"answer": ask_rag(question)}
