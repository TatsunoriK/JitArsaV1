const mongoose = require("mongoose"); // ODM สำหรับเชื่อมต่อ MongoDB

// ============================================================
// UserSchema — เก็บข้อมูลผู้ใช้งานในระบบ
// collection ชื่อ: "users" (Mongoose แปลง plural อัตโนมัติ)
// ใช้คู่กับ authController.js (register / login)
// ============================================================
const UserSchema = new mongoose.Schema({

  // username — ชื่อผู้ใช้สำหรับ login และแสดงใน Sidebar
  // unique: true → MongoDB สร้าง unique index ให้อัตโนมัติ
  //   ถ้าพยายาม insert username ซ้ำ จะ throw error code 11000 (duplicate key)
  //   authController ตรวจด้วย findOne ก่อนเพื่อส่ง message ที่อ่านง่ายกว่า error raw
  // trim: true → ตัด whitespace หัวท้ายออกก่อนบันทึก
  //   กัน edge case เช่น " admin" กับ "admin" ถูกมองว่าต่างกัน
  username: {
    type: String,
    required: true,
    unique: true,
    trim: true,
  },

  // password — เก็บเฉพาะ bcrypt hash เท่านั้น ไม่เคยเก็บ plain text
  // ค่าจะหน้าตาแบบนี้: "$2b$10$N9qo8uLOickgx2ZMRZoMyeIjZAgcfl7p92ldGxad68LJZdL17lhWy"
  // required: true แต่ไม่มี trim เพราะ bcrypt hash ไม่มี whitespace
  // ไม่มี unique เพราะ hash ของรหัสผ่านเดียวกันอาจได้ผลต่างกัน (salt ต่างกัน)
  password: {
    type: String,
    required: true,
  },

  // email — ใช้ตรวจสอบ duplicate ตอนสมัคร และเผื่อส่ง notification ในอนาคต
  // unique: true → ป้องกัน 1 email สมัครได้แค่ 1 account
  // trim: true → กัน " user@email.com" กับ "user@email.com" ถูกมองต่างกัน
  // หมายเหตุ: ยังไม่มี validate format email ในโค้ดนี้
  //   ถ้าอยากตรวจ format ให้เพิ่ม match: [/regex/, "message"] หรือใช้ validator library
  email: {
    type: String,
    required: true,
    unique: true,
    trim: true,
  },

  // createdAt — เวลาที่สมัครสมาชิก บันทึกครั้งเดียวตอน insert ไม่เปลี่ยนอีก
  // default: Date.now — ไม่มี () เพราะต้องเป็น reference ของ function
  //   ถ้าใส่ Date.now() ทุก user จะได้เวลาเดียวกันคือตอนโหลดไฟล์
  // ต่างจาก ChatSessions ที่ใช้ timestamps option เพราะ User ไม่ต้องการ updatedAt
  createdAt: {
    type: Date,
    default: Date.now,
  },
});

// export model ชื่อ "User"
// Mongoose จะ map ไปที่ collection "users" ใน MongoDB อัตโนมัติ
module.exports = mongoose.model("User", UserSchema);