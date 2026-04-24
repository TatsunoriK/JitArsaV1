const ChatSessions = require("../models/ChatSessionsSchema.js");
const ChatMessages = require("../models/ChatMessagesSchema.js");

// GET ALL SESSIONS
exports.getChatHistory = async (req, res) => {
  try {
    const history = await ChatSessions.find()
      .sort({ updated_at: -1 })
      .select("title updated_at created_at");

    res.json({ success: true, data: history });
  } catch (err) {
    console.error(err);
    res.status(500).json({ success: false, message: "เกิดข้อผิดพลาดในการดึงประวัติ" });
  }
};

// GET MESSAGES BY SESSION
exports.getChatMessages = async (req, res) => {
  try {
    const { sessionId } = req.params;
    // ✅ ใช้ session_id (ตรงกับ schema)
    const messages = await ChatMessages.find({ session_id: sessionId })
      .sort({ timestamp: 1 });

    res.json({ success: true, data: messages });
  } catch (err) {
    console.error(err);
    res.status(500).json({ success: false, message: "เกิดข้อผิดพลาดในการดึงข้อความ" });
  }
};

// DELETE SESSION + MESSAGES
exports.deleteChat = async (req, res) => {
  try {
    const { sessionId } = req.params;
    await ChatSessions.findByIdAndDelete(sessionId);
    await ChatMessages.deleteMany({ session_id: sessionId });

    res.json({ success: true, message: "ลบประวัติการสนทนาเรียบร้อยแล้ว" });
  } catch (err) {
    console.error(err);
    res.status(500).json({ success: false });
  }
};
