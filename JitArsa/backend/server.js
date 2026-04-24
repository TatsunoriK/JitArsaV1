const express = require("express");          // web framework หลัก
const cors = require("cors");                // middleware อนุญาต cross-origin request จาก frontend
require("dotenv").config();                 // โหลด .env → process.env (ต้องเรียกก่อนใช้ MONGO_URI, JWT_SECRET)
const connectDB = require("./config/db");   // ฟังก์ชัน connect MongoDB จาก db.js
const mongoose = require("mongoose");       // ODM — ใช้ connect ซ้ำและตรวจ connection error
const auth = require("./controller/authController");       // handler: login, register
const history = require("./controller/historyController"); // handler: getChatHistory, getChatMessages, deleteChat
const ChatMessages = require("./models/ChatMessagesSchema"); // model บันทึกแต่ละข้อความ
const ChatSessions = require("./models/ChatSessionsSchema"); // model บันทึกแต่ละ session

const app = express();
const port = 5000; // Node.js ฟังที่ port 5000, Python FastAPI อยู่ที่ port 8000

// ─── MIDDLEWARE ───────────────────────────────────────────
// cors() อนุญาตทุก origin — frontend React ที่รันอยู่ port อื่นจะเรียก API ได้
app.use(cors());
// express.json() parse body ที่เป็น JSON → req.body (ถ้าไม่มีบรรทัดนี้ req.body = undefined)
app.use(express.json());

// ─── DATABASE ────────────────────────────────────────────
// connectDB() จาก db.js — เชื่อม MongoDB ครั้งแรก (มี error handling และ process.exit ถ้าล้มเหลว)
connectDB();

// connect อีกครั้งตรงๆ เพื่อดัก error แบบ promise-based แยกต่างหาก
// หมายเหตุ: mongoose.connect ถูกเรียก 2 ครั้ง (connectDB + ที่นี่)
// mongoose จัดการ duplicate connection ให้เองโดยไม่เปิด connection ใหม่ซ้ำ
mongoose
  .connect(process.env.MONGO_URI)
  .then(() => console.log("Mongo connected"))
  .catch((err) => console.error("MongoDB connection error:", err.message));

// ============================================================
// POST /ask-pha — route หลักของระบบ
// รับ: { question, history, sessionId } จาก Frontend
// ทำ: บันทึก message → proxy ไป Python → stream คำตอบกลับ → บันทึก AI response
// ============================================================
app.post("/ask-pha", async (req, res) => {
  try {
    const { question, history, sessionId } = req.body;

    // ─── 1. ถอด user_id จาก JWT (optional) ───────────────
    // ถ้าไม่มี token หรือ token ไม่ valid → ใช้ "guest" แทน
    // เพื่อให้ใช้งานได้โดยไม่ต้อง login แต่ประวัติจะไม่ถูกผูกกับ user
    const authHeader = req.headers.authorization;
    let user_id = "guest";
    if (authHeader && authHeader.startsWith("Bearer ")) {
      try {
        const jwt = require("jsonwebtoken");
        // slice(7) ตัด "Bearer " (7 ตัวอักษร) ออก เหลือแค่ตัว token
        const decoded = jwt.verify(
          authHeader.slice(7),
          process.env.JWT_SECRET || "secretkey"
        );
        user_id = decoded.id; // MongoDB _id ของ user
      } catch (_) {
        // token หมดอายุหรือไม่ valid → ปล่อยให้ user_id เป็น "guest" ต่อไป
      }
    }

    // ─── 2. Upsert ChatSession ────────────────────────────
    // สร้าง session ใหม่ถ้ายังไม่มี, ถ้ามีแล้วไม่ทำอะไร
    // title = 50 ตัวแรกของคำถามแรก → แสดงใน Sidebar
    const firstQuestion = question.slice(0, 50);
    try {
      await ChatSessions.findOneAndUpdate(
        { _id: sessionId },        // ค้นหา session ด้วย _id
        {
          $setOnInsert: {          // $setOnInsert = set เฉพาะตอน INSERT ใหม่
            _id: sessionId,        // ถ้า doc มีอยู่แล้ว ไม่แตะอะไรเลย
            user_id,
            title: firstQuestion,
          },
        },
        { upsert: true, returnDocument: "after" }
        // upsert: true = INSERT ถ้าไม่เจอ, UPDATE ถ้าเจอ (แต่ $setOnInsert ทำให้ UPDATE ไม่เปลี่ยนอะไร)
        // returnDocument: 'after' = คืน document หลัง operation (ไม่ได้ใช้ตอนนี้ แต่ดีสำหรับ debug)
      );
      console.log("[SESSION] upserted:", sessionId, "title:", firstQuestion);
    } catch (sessionErr) {
      // session error ไม่ควรหยุด flow ทั้งหมด → log แล้วดำเนินต่อ
      console.error("[SESSION ERROR]", sessionErr.message);
    }

    // ─── 3. บันทึก user message ลง DB ────────────────────
    // บันทึกก่อนส่งไป Python เพื่อให้มี record แม้ Python จะล้มเหลว
    await ChatMessages.create({
      session_id: sessionId,
      role: "user",
      content: question,
    });

    console.log("[DEBUG] question:", question);
    console.log("[DEBUG] history length:", (history || []).length);

    // ─── 4. Sanitize history ──────────────────────────────
    // กรองเฉพาะ message ที่มีทั้ง role และ content
    // แปลงทุกค่าเป็น String เพื่อกัน type error ใน Python
    const safeHistory = (history || [])
      .filter((h) => h && h.role && h.content)
      .map((h) => ({ role: String(h.role), content: String(h.content) }));

    // ─── 5. AbortController timeout ──────────────────────
    // ถ้า Python ไม่ตอบใน 60 วินาที → abort fetch → throw AbortError
    // กัน request ค้างอยู่นาน (เช่น Python crash หรือ Groq timeout)
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 60000);

    // ─── 6. Proxy request → Python FastAPI ───────────────
    // ส่ง question + history ไปยัง main.py ที่ port 8000
    // ไม่ส่ง sessionId เพราะ Python ไม่ต้องรู้ว่าเป็น session ไหน
    const response = await fetch("http://localhost:8000/ask-pha", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history: safeHistory }),
      signal: controller.signal, // เชื่อมกับ AbortController
    });

    // ยกเลิก timeout เพราะ Python ตอบแล้ว (กัน abort หลัง response มาแล้ว)
    clearTimeout(timeout);

    if (!response.ok) {
      const text = await response.text();
      console.error("Python error:", text);
      return res.status(500).json({ error: "Python backend error" });
    }

    // ─── 7. ตั้งค่า response headers สำหรับ streaming ────
    // ลบ Content-Length เพราะไม่รู้ขนาดล่วงหน้า (streaming ไม่รู้ความยาว)
    res.removeHeader("Content-Length");
    // text/plain; charset=utf-8 — browser จะแสดงผลทันทีที่ได้ chunk
    res.setHeader("Content-Type", "text/plain; charset=utf-8");
    // X-Accel-Buffering: no — บอก Nginx ให้ไม่ buffer response (ส่งทันที)
    res.setHeader("X-Accel-Buffering", "no");
    // Cache-Control: no-cache — บอก browser/proxy ไม่ต้อง cache streaming response
    res.setHeader("Cache-Control", "no-cache");

    // ─── 8. Pipe stream Python → Browser ─────────────────
    // อ่าน response จาก Python ทีละ chunk แล้วส่งต่อให้ browser ทันที
    // ผู้ใช้เห็น text ไหลออกมาทีละนิดแบบ real-time
    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let fullAiResponse = ""; // สะสม text ทั้งหมดเพื่อบันทึกลง DB ตอนท้าย

    while (true) {
      const { done, value } = await reader.read();
      if (done) break; // Python stream จบแล้ว
      // stream: true = บอก decoder ว่ายังมี chunk ถัดไป (กัน multi-byte char ขาดกลาง chunk)
      const chunk = decoder.decode(value, { stream: true });
      fullAiResponse += chunk;
      res.write(chunk); // ส่ง chunk ให้ browser ทันทีโดยไม่รอ stream จบ
    }

    // ─── 9. บันทึก AI response ลง DB ─────────────────────
    // บันทึกหลัง stream จบเท่านั้น เพราะต้องการ response เต็มๆ
    // ถ้าบันทึกทีละ chunk จะได้ row แยกเป็นพันๆ rows แทนที่จะเป็น 1 row
    await ChatMessages.create({
      session_id: sessionId,
      role: "assistant",
      content: fullAiResponse,
    });

    res.end(); // ปิด response stream อย่างถูกต้อง
  } catch (error) {
    console.error("FULL ERROR:", error);
    if (error.name === "AbortError") {
      // AbortController.abort() ถูกเรียกหลัง 60 วินาที
      return res
        .status(504) // 504 Gateway Timeout
        .json({ error: "Python ใช้เวลานานเกินไป (timeout)" });
    }
    res.status(500).json({ error: "Connection failed", detail: error.message });
  }
});

// ─── GUARD ROUTE ─────────────────────────────────────────
// ป้องกัน browser เปิด /ask-pha ตรงๆ แล้วเจอ error แปลกๆ
app.get("/ask-pha", (req, res) => {
  res.send("Use POST method instead");
});

// ─── AUTH ROUTES ─────────────────────────────────────────
app.post("/login", auth.login);        // authController.login
app.post("/register", auth.register);  // authController.register

// ─── HISTORY ROUTES ──────────────────────────────────────
app.get("/api/history", history.getChatHistory);               // ดึง session ทั้งหมดของ user (ต้องมี token)
app.get("/api/history/:sessionId", history.getChatMessages);   // ดึง messages ใน session
app.delete("/api/history/:sessionId", history.deleteChat);     // ลบ session + messages

// ─── START SERVER ─────────────────────────────────────────
app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});