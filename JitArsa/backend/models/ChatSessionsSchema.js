const mongoose = require("mongoose");

const ChatSessionsSchema = new mongoose.Schema(
{
    _id: {
        type: String,
        required: true
    },
    user_id: {
        type: String,
        required: true,
        index: true
    },
    title: {
        type: String,
        default: "New Chat"
    }
},
{
    timestamps: {
        createdAt: "created_at",
        updatedAt: "updated_at"
    }
}
);

module.exports = mongoose.model("ChatSessions", ChatSessionsSchema);
