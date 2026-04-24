const ChatSessions = require("../models/ChatSessionsSchema.js"); // model สำหรับ collection chat_sessions
const ChatMessages = require("../models/ChatMessagesSchema.js"); // model สำหรับ collection chat_messages
const jwt = require("jsonwebtoken"); // ใช้ถอดรหัส JWT token เพื่อดึง user_id

// ============================================================
// HELPER — ดึง user_id จาก Authorization header
// ใช้ร่วมกันในทุก route ที่ต้องการรู้ว่าใครเรียก
// คืน: user_id (string) ถ้า token ถูกต้อง, null ถ้าไม่มีหรือ token หมดอายุ
// ============================================================
function getUserIdFromReq(req) {
  // Authorization header รูปแบบ: "Bearer <token>"
  const authHeader = req.headers.authorization;

  // ถ้าไม่มี header หรือไม่ได้ขึ้นต้นด้วย "Bearer " → ถือว่าไม่ได้ login
  if (!authHeader || !authHeader.startsWith("Bearer ")) return null;

  try {
    // slice(7) ตัด "Bearer " (7 ตัวอักษร) ออก เหลือแค่ตัว token
    // jwt.verify ถอดรหัส token + ตรวจสอบ signature และวันหมดอายุในคราวเดียว
    // ถ้า token ถูกแก้ไขหรือหมดอายุ จะ throw error เข้า catch ทันที
    const decoded = jwt.verify(
      authHeader.slice(7),
      process.env.JWT_SECRET || "secretkey"
    );

    // decoded.id คือ user._id ที่ฝังไว้ตอน jwt.sign ใน authController
    return decoded.id;
  } catch (_) {
    // token ไม่ valid (หมดอายุ, ถูกแก้ไข, หรือ sign ด้วย secret ต่างกัน)
    return null;
  }
}

// ============================================================
// GET /api/history — ดึงรายการ session ทั้งหมดของ user คนนี้
// ใช้แสดงใน Sidebar ฝั่ง Frontend
// ============================================================
exports.getChatHistory = async (req, res) => {
  try {
    // ถอด user_id จาก token ก่อนทำอะไรทั้งนั้น
    const user_id = getUserIdFromReq(req);

    // ถ้า user_id เป็น null = ไม่ได้ login หรือ token หมดอายุ → บังคับ login
    if (!user_id) {
      return res.status(401).json({ success: false, message: "กรุณา login ก่อน" });
    }

    // สำคัญ: filter { user_id } ทุกครั้ง
    // ถ้าไม่มีบรรทัดนี้ จะดึง session ของทุกคนในระบบออกมา
    const history = await ChatSessions.find({ user_id })
      .sort({ updated_at: -1 })          // เรียงล่าสุดขึ้นก่อน (Sidebar แสดงตามลำดับนี้)
      .select("title updated_at created_at"); // ดึงเฉพาะ field ที่ต้องใช้ ไม่ดึง user_id มาเปลือง bandwidth

    res.json({ success: true, data: history });
  } catch (err) {
    console.error(err);
    res.status(500).json({ success: false, message: "เกิดข้อผิดพลาดในการดึงประวัติ" });
  }
};

// ============================================================
// GET /api/history/:sessionId — ดึง messages ทั้งหมดใน session นั้น
// ใช้ตอนเปิดหน้า ChatHistory.jsx เพื่อโหลดบทสนทนาเก่า
// ============================================================
exports.getChatMessages = async (req, res) => {
  try {
    // sessionId มาจาก URL parameter เช่น /api/history/abc-123
    const { sessionId } = req.params;

    // ไม่ต้อง filter user_id ที่นี่ เพราะ sessionId เป็น UUID ที่ random มาก
    // โอกาสเดาถูกแทบเป็นไปไม่ได้ และ session ถูกผูกกับ user อยู่แล้วตอนสร้าง
    const messages = await ChatMessages.find({ session_id: sessionId })
      .sort({ timestamp: 1 }); // เรียงเก่าสุดขึ้นก่อน → แสดงบทสนทนาตามลำดับเวลา

    res.json({ success: true, data: messages });
  } catch (err) {
    console.error(err);
    res.status(500).json({ success: false, message: "เกิดข้อผิดพลาดในการดึงข้อความ" });
  }
};

// ============================================================
// DELETE /api/history/:sessionId — ลบ session พร้อม messages ทั้งหมด
// เรียกจากปุ่มลบใน Sidebar หลังผู้ใช้กด confirm
// ============================================================
exports.deleteChat = async (req, res) => {
  try {
    const { sessionId } = req.params;

    // ลบ session document ออกจาก chat_sessions
    // findByIdAndDelete ใช้ _id ตรงๆ (sessionId คือ _id ของ ChatSession)
    await ChatSessions.findByIdAndDelete(sessionId);

    // ลบ messages ทุกตัวที่อยู่ใน session นี้
    // deleteMany ลบทีเดียวหลายตัว ดีกว่า loop deleteOne
    // ต้องทำทั้งสองขั้นตอน ไม่งั้น messages กำพร้าค้างอยู่ใน DB โดยไม่มี session
    await ChatMessages.deleteMany({ session_id: sessionId });

    res.json({ success: true, message: "ลบประวัติการสนทนาเรียบร้อยแล้ว" });
  } catch (err) {
    console.error(err);
    res.status(500).json({ success: false });
  }
};