// const UserSchema = new mongoose.Schema({
//   name: String,
//   email: { type: String, unique: true },
//   createdAt: { type: Date, default: Date.now }
// });

// const User = mongoose.model('User', UserSchema);

const mongoose = require("mongoose");

const UserSchema = new mongoose.Schema({
  username: {
    type: String,
    required: true,
    unique: true,
    trim: true,
  },
  password: {
    type: String,
    required: true,
  },
  email: {
    type: String,
    required: true,
    unique: true,
    trim: true,
  },
  createdAt: {
    type: Date,
    default: Date.now,
  },
});

module.exports = mongoose.model("User", UserSchema);
