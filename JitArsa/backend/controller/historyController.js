const ChatSessions = require("../models/ChatSessionsSchema.js");
const ChatMessages = require("../models/ChatMessagesSchema.js");
const jwt = require("jsonwebtoken");

// helper: ดึง user_id จาก Authorization header
function getUserIdFromReq(req) {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith("Bearer ")) return null;
  try {
    const decoded = jwt.verify(
      authHeader.slice(7),
      process.env.JWT_SECRET || "secretkey"
    );
    return decoded.id;
  } catch (_) {
    return null;
  }
}

// GET SESSIONS ของ user คนนี้เท่านั้น
exports.getChatHistory = async (req, res) => {
  try {
    const user_id = getUserIdFromReq(req);
    if (!user_id) {
      return res.status(401).json({ success: false, message: "กรุณา login ก่อน" });
    }

    // ✅ filter ตาม user_id
    const history = await ChatSessions.find({ user_id })
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