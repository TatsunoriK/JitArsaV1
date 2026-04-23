const mongoose = require("mongoose");

const ChatMessagesSchema = new mongoose.Schema(
{
    session_id: {
        type: String,
        required: true,
        ref: "ChatSessions",
        index: true
    },
    role: {
        type: String,
        enum: ["user", "assistant"],
        required: true
    },
    content: {
        type: String,
        required: true
    },
    timestamp: {
        type: Date,
        default: Date.now
    }
}
);

module.exports = mongoose.model("ChatMessages", ChatMessagesSchema);