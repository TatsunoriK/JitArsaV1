const express = require("express");
const bcrypt = require("bcrypt");   // ใช้ hash และ verify รหัสผ่าน (one-way encryption)
const jwt = require("jsonwebtoken"); // ใช้สร้างและตรวจสอบ JWT token
const User = require("../models/UserSchema"); // Mongoose model สำหรับ collection "users"

// ============================================================
// REGISTER — สมัครสมาชิกใหม่
// รับ: POST /register  body: { username, password, email }
// ============================================================
exports.register = async (req, res) => {
  try {
    // ดึงข้อมูลจาก body ที่ client ส่งมา
    const { username, password, email } = req.body;

    // ตรวจว่ามี username นี้ในระบบแล้วหรือยัง
    // findOne คืน null ถ้าไม่เจอ, คืน document ถ้าเจอ
    const exist = await User.findOne({ username });
    if (exist) {
      // username ซ้ำ → ส่ง success: false กลับ (ไม่ใช่ error 4xx เพื่อให้ client จัดการง่าย)
      return res.json({
        success: false,
        message: "มีผู้ใช้นี้แล้ว",
      });
    }

    // ตรวจว่ามี email นี้ในระบบแล้วหรือยัง (email ต้อง unique เช่นกัน)
    const existEmail = await User.findOne({ email });
    if (existEmail) {
      return res.json({
        success: false,
        message: "อีเมลนี้ถูกใช้แล้ว",
      });
    }

    // Hash รหัสผ่านก่อนเก็บลง DB — ห้ามเก็บ plain text เด็ดขาด
    // saltRounds = 10 หมายความว่า bcrypt จะ hash 2^10 = 1,024 รอบ
    // ยิ่งมาก = ยิ่งปลอดภัย แต่ยิ่งช้า, ค่า 10 เป็น standard ที่นิยม
    const hashedPassword = await bcrypt.hash(password, 10);

    // สร้าง document ใหม่ใน collection users
    const user = new User({
      username,
      password: hashedPassword, // เก็บเฉพาะ hash ไม่เก็บ password ดิบ
      email,
    });

    // บันทึกลง MongoDB
    await user.save();

    // ส่ง response สำเร็จกลับ (ไม่ส่ง token ตอนสมัคร ให้ไป login เอง)
    res.json({
      success: true,
      message: "สมัครสำเร็จ",
    });
  } catch (err) {
    // error ที่ไม่คาดคิด เช่น DB timeout, validation fail จาก Mongoose schema
    console.error(err);
    res.status(500).json({ success: false });
  }
};

// ============================================================
// LOGIN — เข้าสู่ระบบ
// รับ: POST /login  body: { username, password }
// คืน: { success, token, user: { id, username } }
// ============================================================
exports.login = async (req, res) => {
  try {
    const { username, password } = req.body;

    // ค้นหา user จาก username ใน DB
    const user = await User.findOne({ username });
    if (!user) {
      // ไม่พบ username นี้เลยในระบบ
      return res.json({
        success: false,
        message: "ไม่พบผู้ใช้",
      });
    }

    // เปรียบเทียบ password ที่พิมพ์มา กับ hash ที่เก็บใน DB
    // bcrypt.compare จะ hash password ด้วย salt ที่ฝังใน hash เดิม แล้วเทียบ
    // คืน true = ตรงกัน, false = ไม่ตรง
    const isMatch = await bcrypt.compare(password, user.password);

    if (!isMatch) {
      // password ไม่ตรงกับที่ hash ไว้
      return res.json({
        success: false,
        message: "รหัสผ่านไม่ถูกต้อง",
      });
    }

    // สร้าง JWT token
    // payload: { id, username } — ข้อมูลที่ฝังใน token (อ่านได้ แต่แก้ไม่ได้)
    // secret: ใช้ตรวจสอบว่า token ไม่ถูกปลอมแปลง
    // expiresIn: "1d" — token หมดอายุใน 1 วัน หลังจากนั้นต้อง login ใหม่
    const token = jwt.sign(
      { id: user._id, username: user.username },
      process.env.JWT_SECRET || "secretkey", // ควรตั้งค่าใน .env ให้เสมอ
      { expiresIn: "1d" }
    );

    // ส่ง token และข้อมูล user พื้นฐานกลับไป
    // client จะเก็บ token ไว้ใน localStorage แล้วแนบทุก request ที่ต้องการ auth
    res.json({
      success: true,
      token,
      user: {
        id: user._id,
        username: user.username,
        // ไม่ส่ง password (แม้จะเป็น hash) กลับไปเด็ดขาด
      },
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ success: false });
  }
};