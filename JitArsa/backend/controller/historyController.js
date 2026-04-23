const express = require("express");
const ChatSessions = require("../models/ChatSessionsSchema.js");
const ChatMessages = require("../models/ChatMessagesSchema.js");

// GET ALL SESSIONS
exports.getChatHistory = async (req, res) => {
  try {
    const history = await History.find()
      .sort({ updatedAt: -1 }) // ล่าสุดอยู่บน
      .select("title updatedAt");

    res.json({
      success: true,
      data: history,
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ success: false, message: "เกิดข้อผิดพลาดในการดึงประวัติ" });
  }
};

// GET MESSAGES
exports.getChatMessages = async (req, res) => {
  try {
    const { sessionId } = req.params;

    const messages = await Message.find({ sessionId })
      .sort({ timestamp: 1 }); // เรียงตามเวลาส่งจากเก่าไปใหม่

    res.json({
      success: true,
      data: messages,
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ success: false, message: "เกิดข้อผิดพลาดในการดึงข้อความ" });
  }
};

// DELETE SESSION
exports.deleteChat = async (req, res) => {
  try {
    const { sessionId } = req.params;
    
    await History.findByIdAndDelete(sessionId);
    await Message.deleteMany({ sessionId }); 

    res.json({
      success: true,
      message: "ลบประวัติการสนทนาเรียบร้อยแล้ว",
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ success: false });
  }
};