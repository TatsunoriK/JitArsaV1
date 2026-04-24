const mongoose = require("mongoose"); // ODM (Object Document Mapper) สำหรับเชื่อมต่อ MongoDB

// ============================================================
// ChatMessagesSchema — เก็บแต่ละข้อความในบทสนทนา
// 1 session มีได้หลาย message (one-to-many กับ ChatSessions)
// collection ชื่อ: "chatmessages" (Mongoose แปลง plural อัตโนมัติ)
// ============================================================
const ChatMessagesSchema = new mongoose.Schema({

  // session_id — บอกว่า message นี้อยู่ใน session ไหน
  // ใช้ String แทน ObjectId เพราะ sessionId สร้างจาก crypto.randomUUID() ฝั่ง Frontend
  // ref: "ChatSessions" บอก Mongoose ว่า field นี้ reference ไปที่ collection ChatSessions
  //   → สามารถใช้ .populate("session_id") เพื่อดึงข้อมูล session มาได้ในอนาคต
  // index: true สร้าง index บน field นี้ใน MongoDB
  //   → ทำให้ ChatMessages.find({ session_id: "..." }) เร็วขึ้นมาก
  //   → สำคัญมากเพราะ query นี้ถูกเรียกทุกครั้งที่เปิดหน้า ChatHistory
  session_id: {
    type: String,
    required: true,
    ref: "ChatSessions",
    index: true,
  },

  // role — บอกว่าข้อความนี้มาจากใคร
  // enum จำกัดค่าให้รับได้แค่ "user" หรือ "assistant" เท่านั้น
  // ถ้าส่งค่าอื่นมา Mongoose จะ throw ValidationError ก่อนบันทึก
  // "assistant" คือชื่อมาตรฐานของ OpenAI/Groq API format
  // (Frontend ใช้ "bot" แต่ตอนส่งมา server แปลงเป็น "assistant" ก่อนเสมอ)
  role: {
    type: String,
    enum: ["user", "assistant"],
    required: true,
  },

  // content — เนื้อหาข้อความ ทั้งฝั่ง user และ AI
  // ไม่จำกัดความยาว (String ใน MongoDB รองรับได้ถึง 16MB)
  content: {
    type: String,
    required: true,
  },

  // timestamp — เวลาที่สร้าง message นี้
  // default: Date.now ใส่เวลาปัจจุบันให้อัตโนมัติตอน create
  // หมายเหตุ: เขียน Date.now ไม่มี () — ถ้าใส่ () จะ evaluate ตอนโหลดไฟล์
  //   ทุก document จะได้เวลาเดียวกัน ต้องเป็น reference ของ function ไม่ใช่ค่า
  // ใช้ sort({ timestamp: 1 }) ใน historyController เพื่อเรียงข้อความตามเวลา
  timestamp: {
    type: Date,
    default: Date.now,
  },
});

// export model ชื่อ "ChatMessages"
// Mongoose จะ map ไปที่ collection "chatmessages" ใน MongoDB อัตโนมัติ
module.exports = mongoose.model("ChatMessages", ChatMessagesSchema);