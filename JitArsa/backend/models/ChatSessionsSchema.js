const mongoose = require("mongoose"); // ODM สำหรับเชื่อมต่อ MongoDB

// ============================================================
// ChatSessionsSchema — เก็บข้อมูลแต่ละ session การสนทนา
// 1 user มีได้หลาย session (one-to-many กับ User)
// 1 session มีได้หลาย message (one-to-many กับ ChatMessages)
// collection ชื่อ: "chatsessions" (Mongoose แปลง plural อัตโนมัติ)
// ============================================================
const ChatSessionsSchema = new mongoose.Schema(
  {
    // _id — override ค่า default ของ Mongoose ที่ปกติจะสร้าง ObjectId ให้อัตโนมัติ
    // ใช้ String แทนเพราะ sessionId สร้างจาก crypto.randomUUID() ฝั่ง Frontend
    // ตัวอย่าง: "550e8400-e29b-41d4-a716-446655440000"
    // ข้อดี: Frontend รู้ sessionId ก่อนส่ง request ครั้งแรก ไม่ต้องรอให้ DB generate
    _id: {
      type: String,
      required: true,
    },

    // user_id — บอกว่า session นี้เป็นของ user คนไหน
    // ค่ามาจาก decoded.id ที่ถอดออกมาจาก JWT token ใน server.js
    // String เพราะ MongoDB ObjectId ถูกแปลงเป็น string ตอน jwt.sign
    // index: true สร้าง index บน field นี้
    //   → ทำให้ ChatSessions.find({ user_id }) ใน historyController เร็วขึ้นมาก
    //   → สำคัญเพราะทุกครั้งที่ Sidebar โหลด จะ query ด้วย user_id เสมอ
    user_id: {
      type: String,
      required: true,
      index: true,
    },

    // title — ชื่อย่อของบทสนทนา แสดงใน Sidebar
    // ได้มาจาก 50 ตัวอักษรแรกของคำถามแรกในบทสนทนา
    // ตั้งค่าผ่าน $setOnInsert ใน server.js ตอน upsert session ครั้งแรก
    // default: "New Chat" รองรับกรณีที่สร้าง session โดยไม่มี title (edge case)
    title: {
      type: String,
      default: "New Chat",
    },
  },

  // ============================================================
  // Schema Options — ตัวเลือกเพิ่มเติมของ Schema (argument ที่ 2)
  // ============================================================
  {
    // timestamps: true ให้ Mongoose จัดการ created_at และ updated_at ให้อัตโนมัติ
    // - created_at: ตั้งค่าตอน insert ครั้งแรก ไม่เปลี่ยนอีก
    // - updated_at: อัปเดตทุกครั้งที่มีการแก้ไข document
    // ใช้ rename เพื่อให้ชื่อ field เป็น snake_case แทน camelCase default (createdAt/updatedAt)
    // historyController ใช้ .sort({ updated_at: -1 }) เพื่อเรียง session ล่าสุดขึ้นก่อน
    timestamps: {
      createdAt: "created_at",
      updatedAt: "updated_at",
    },
  }
);

// export model ชื่อ "ChatSessions"
// Mongoose จะ map ไปที่ collection "chatsessions" ใน MongoDB อัตโนมัติ
module.exports = mongoose.model("ChatSessions", ChatSessionsSchema);