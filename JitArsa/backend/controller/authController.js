const express = require("express");
const bcrypt = require("bcrypt");
const jwt = require("jsonwebtoken");
const User = require("../models/UserSchema");

// REGISTER
exports.register = async (req, res) => {
  try {
    const { username, password, email } = req.body;

    const exist = await User.findOne({ username });
    if (exist) {
      return res.json({
        success: false,
        message: "มีผู้ใช้นี้แล้ว",
      });
    }

    const existEmail = await User.findOne({ email });
    if (existEmail) {
      return res.json({
        success: false,
        message: "อีเมลนี้ถูกใช้แล้ว",
      });
    }

    const hashedPassword = await bcrypt.hash(password, 10);

    const user = new User({
      username,
      password: hashedPassword,
      email,
    });

    await user.save();

    res.json({
      success: true,
      message: "สมัครสำเร็จ",
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ success: false });
  }
};

// LOGIN
exports.login = async (req, res) => {
  try {
    const { username, password } = req.body;

    const user = await User.findOne({ username });
    if (!user) {
      return res.json({
        success: false,
        message: "ไม่พบผู้ใช้",
      });
    }

    const isMatch = await bcrypt.compare(password, user.password);

    if (!isMatch) {
      return res.json({
        success: false,
        message: "รหัสผ่านไม่ถูกต้อง",
      });
    }

    // ✅ ใช้ process.env.JWT_SECRET แทน hardcode
    const token = jwt.sign(
      { id: user._id, username: user.username },
      process.env.JWT_SECRET || "secretkey",
      { expiresIn: "1d" },
    );

    res.json({
      success: true,
      token,
      user: {
        id: user._id,
        username: user.username,
      },
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ success: false });
  }
};
