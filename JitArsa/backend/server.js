const express = require("express");
const mongoose = require("mongoose");
require("dotenv").config();

const app = express();
const port = 5000;

app.use(express.json());

app.post("/ask-pha", async (req, res) => {
  try {
    const { question, history } = req.body;

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 60000);

    const response = await fetch("http://localhost:8000/ask-pha", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history }),
      signal: controller.signal,
    });

    clearTimeout(timeout);

    if (!response.ok) {
      const text = await response.text();
      console.error("Python error:", text);
      return res.status(500).json({ error: "Python backend error" });
    }

    // Always stream through — Python always returns text/plain streaming
    res.setHeader("Content-Type", "text/plain; charset=utf-8");
    res.setHeader("Transfer-Encoding", "chunked");
    res.setHeader("X-Accel-Buffering", "no");

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
      return res.status(504).json({ error: "Python ใช้เวลานานเกินไป (timeout)" });
    }
    res.status(500).json({ error: "Connection failed", detail: error.message });
  }
});

app.get("/ask-pha", (req, res) => {
  res.send("Use POST method instead");
});

mongoose.connect(process.env.MONGO_URI)
  .then(() => {
    console.log("Mongo connected");
    app.listen(port, () => {
      console.log(`Server running at http://localhost:${port}`);
    });
  })
  .catch((err) => {
    console.error("MongoDB connection error:", err.message);
  });