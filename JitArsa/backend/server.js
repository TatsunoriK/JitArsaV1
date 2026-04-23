const express = require("express");
const cors = require("cors");
require("dotenv").config();
const connectDB = require("./config/db");

const mongoose = require("mongoose");
const auth = require("./controller/authController");

const app = express();
const port = 5000;

app.use(cors());
app.use(express.json());

connectDB();

app.post("/ask-pha", async (req, res) => {
  try {
    const { question, history } = req.body;

    // DEBUG: ตรวจว่า frontend ส่ง history มาครบไหม
    console.log("[DEBUG] question:", question);
    console.log("[DEBUG] history length:", (history || []).length);
    (history || []).forEach((h, i) => {
      console.log(`[DEBUG]   history[${i}] role=${h.role} content=${String(h.content).slice(0, 60)}`);
    });

    // ตรวจสอบ history ให้ถูกรูปแบบก่อนส่งต่อ
    const safeHistory = (history || [])
      .filter(h => h && h.role && h.content)
      .map(h => ({
        role: String(h.role),
        content: String(h.content),
      }));

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 60000);

    const response = await fetch("http://localhost:8000/ask-pha", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history: safeHistory }),
      signal: controller.signal,
    });

    clearTimeout(timeout);

    if (!response.ok) {
      const text = await response.text();
      console.error("Python error:", text);
      return res.status(500).json({ error: "Python backend error" });
    }

    // ลบ Content-Length ถ้า Express ใส่มาอัตโนมัติ (conflict กับ Transfer-Encoding)
    res.removeHeader("Content-Length");
    res.setHeader("Content-Type", "text/plain; charset=utf-8");
    res.setHeader("X-Accel-Buffering", "no");
    res.setHeader("Cache-Control", "no-cache");
    // ไม่ set Transfer-Encoding เอง — Node จัดการให้อัตโนมัติเมื่อ res.write()

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      res.write(decoder.decode(value, { stream: true }));
    }
    res.end();
  } catch (error) {
    console.error("FULL ERROR:", error);
    if (error.name === "AbortError") {
      return res
        .status(504)
        .json({ error: "Python ใช้เวลานานเกินไป (timeout)" });
    }
    res.status(500).json({ error: "Connection failed", detail: error.message });
  }
});

app.get("/ask-pha", (req, res) => {
  res.send("Use POST method instead");
});

app.post("/login", auth.login);
app.post("/register", auth.register);

mongoose
  .connect(process.env.MONGO_URI)
  .then(() => {
    console.log("Mongo connected");
    app.listen(port, () => {
      console.log(`Server running at http://localhost:${port}`);
    });
  })
  .catch((err) => {
    console.error("MongoDB connection error:", err.message);
  });
