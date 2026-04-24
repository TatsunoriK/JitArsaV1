const express = require("express");
const cors = require("cors");
require("dotenv").config();
const connectDB = require("./config/db");
const mongoose = require("mongoose");
const auth = require("./controller/authController");
const history = require("./controller/historyController");
const ChatMessages = require("./models/ChatMessagesSchema");
const ChatSessions = require("./models/ChatSessionsSchema");

const app = express();
const port = 5000;

app.use(cors());
app.use(express.json());

connectDB();

mongoose
  .connect(process.env.MONGO_URI)
  .then(() => console.log("Mongo connected"))
  .catch((err) => console.error("MongoDB connection error:", err.message));

// POST /ask-pha
app.post("/ask-pha", async (req, res) => {
  try {
    const { question, history, sessionId } = req.body;

    // ดึง user_id จาก token (ถ้ามี) หรือใช้ "guest"
    const authHeader = req.headers.authorization;
    let user_id = "guest";
    if (authHeader && authHeader.startsWith("Bearer ")) {
      try {
        const jwt = require("jsonwebtoken");
        const decoded = jwt.verify(
          authHeader.slice(7),
          process.env.JWT_SECRET || "secretkey"
        );
        user_id = decoded.id;
      } catch (_) {}
    }

    // ✅ สร้าง ChatSession ถ้ายังไม่มี (upsert)
    const firstQuestion = question.slice(0, 50);
    try {
      await ChatSessions.findOneAndUpdate(
        { _id: sessionId },
        {
          $setOnInsert: {
            _id: sessionId,
            user_id,
            title: firstQuestion,
          },
        },
        { upsert: true, returnDocument: 'after' }
      );
      console.log("[SESSION] upserted:", sessionId, "title:", firstQuestion);
    } catch (sessionErr) {
      console.error("[SESSION ERROR]", sessionErr.message);
    }

    await ChatMessages.create({
      session_id: sessionId,
      role: "user",
      content: question,
    });

    console.log("[DEBUG] question:", question);
    console.log("[DEBUG] history length:", (history || []).length);

    const safeHistory = (history || [])
      .filter((h) => h && h.role && h.content)
      .map((h) => ({ role: String(h.role), content: String(h.content) }));

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

    res.removeHeader("Content-Length");
    res.setHeader("Content-Type", "text/plain; charset=utf-8");
    res.setHeader("X-Accel-Buffering", "no");
    res.setHeader("Cache-Control", "no-cache");

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let fullAiResponse = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      fullAiResponse += chunk;
      res.write(chunk);
    }

    await ChatMessages.create({
      session_id: sessionId,
      role: "assistant",
      content: fullAiResponse,
    });

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

app.get("/api/history", history.getChatHistory);
app.get("/api/history/:sessionId", history.getChatMessages);
app.delete("/api/history/:sessionId", history.deleteChat);

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
