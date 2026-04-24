import { Link, useNavigate } from "react-router-dom";
import { useState, useEffect, useRef } from "react";

// จัด group ประวัติตามวัน แบบ ChatGPT
function groupByDate(chats) {
  const now = new Date();
  const groups = { "วันนี้": [], "เมื่อวาน": [], "7 วันที่แล้ว": [], "เก่ากว่านั้น": [] };

  chats.forEach((chat) => {
    const d = new Date(chat.updated_at || chat.created_at || 0);
    const diffDays = Math.floor((now - d) / (1000 * 60 * 60 * 24));
    if (diffDays < 1) groups["วันนี้"].push(chat);
    else if (diffDays < 2) groups["เมื่อวาน"].push(chat);
    else if (diffDays < 7) groups["7 วันที่แล้ว"].push(chat);
    else groups["เก่ากว่านั้น"].push(chat);
  });

  return Object.entries(groups).filter(([, items]) => items.length > 0);
}

function Sidebar({ onNewChat, onRegisterRefresh }) {
  const token = localStorage.getItem("jp_token");
  const username = localStorage.getItem("jp_username");
  const [chatHistory, setChatHistory] = useState([]);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const navigate = useNavigate();

  const fetchHistoryRef = useRef(null);
  fetchHistoryRef.current = async () => {
    const t = localStorage.getItem("jp_token");
    if (!t) return;
    setLoadingHistory(true);
    try {
      const response = await fetch("/api/history", {
        headers: { Authorization: `Bearer ${t}` },
      });
      const result = await response.json();
      if (result.success) {
        setChatHistory(result.data);
      }
    } catch (error) {
      console.error("Error fetching history:", error);
    } finally {
      setLoadingHistory(false);
    }
  };

  useEffect(() => {
    fetchHistoryRef.current();
  }, []);

  useEffect(() => {
    if (onRegisterRefresh) {
      onRegisterRefresh(() => fetchHistoryRef.current());
    }
  }, [onRegisterRefresh]);

  const handleNewChat = () => {
    if (onNewChat) {
      onNewChat();
      setTimeout(() => fetchHistoryRef.current(), 300);
    } else {
      navigate("/");
      window.location.reload();
    }
  };

  const currentSessionId = sessionStorage.getItem("jp_session_id");
  const grouped = groupByDate(chatHistory);

  return (
    <>
      <style>
        {`
        .custom-sidebar::-webkit-scrollbar { width: 4px; }
        .custom-sidebar::-webkit-scrollbar-thumb {
          background: rgba(172, 177, 214, 0.4);
          border-radius: 10px;
        }
        .chat-item-active {
          background: rgba(24, 95, 165, 0.12) !important;
        }
        `}
      </style>

      <div
        className="w-56 h-screen flex flex-col px-3 py-4 border-r"
        style={{
          background: "rgba(255,255,255,0.6)",
          backdropFilter: "blur(16px)",
          WebkitBackdropFilter: "blur(16px)",
          borderRight: "1px solid rgba(172,177,214,0.4)",
        }}
      >
        {/* Header */}
        <div className="flex items-center gap-2 mb-3">
          <img src="/girl.png" alt="bot" className="w-8 h-8 rounded-full object-cover border-2 border-[#ACB1D6]/40" />
          <h4 className="text-sm font-semibold" style={{ color: "#185fa5" }}>Nop Napha</h4>
        </div>

        {/* ปุ่ม New Chat */}
        <button
          onClick={handleNewChat}
          className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-xl text-sm font-medium mb-3 transition-all hover:opacity-80 active:scale-95"
          style={{
            background: "linear-gradient(135deg, #8294C4, #6b7fbb)",
            color: "#fff",
            border: "none",
            boxShadow: "0 4px 12px rgba(130,148,196,0.3)",
          }}
        >
          ✏️ New Chat
        </button>

        {/* รายการประวัติ แบบ ChatGPT */}
        <div className="flex-1 overflow-y-auto pr-1 custom-sidebar">
          {loadingHistory ? (
            <div className="px-2 py-4 text-xs text-gray-400 text-center">กำลังโหลด...</div>
          ) : grouped.length > 0 ? (
            grouped.map(([label, chats]) => (
              <div key={label} className="mb-3">
                {/* Group label */}
                <p className="text-[10px] font-semibold px-2 mb-1 uppercase tracking-widest"
                  style={{ color: "#9ca3af" }}>
                  {label}
                </p>
                {chats.map((chat) => {
                  const isActive = chat._id === currentSessionId;
                  return (
                    <Link
                      key={chat._id}
                      to={`/chat/${chat._id}`}
                      className={`flex items-center gap-2 px-2 py-2 rounded-lg text-xs transition-all no-underline hover:no-underline group mb-0.5 ${isActive ? "chat-item-active" : ""}`}
                      style={{ color: isActive ? "#185fa5" : "#4a5568" }}
                      onMouseEnter={(e) => {
                        if (!isActive) e.currentTarget.style.backgroundColor = "rgba(24,95,165,0.06)";
                      }}
                      onMouseLeave={(e) => {
                        if (!isActive) e.currentTarget.style.backgroundColor = "transparent";
                      }}
                    >
                      <i className={`bi bi-chat-text text-xs flex-shrink-0 ${isActive ? "text-[#185fa5]" : "opacity-30 group-hover:opacity-70"}`}></i>
                      <span className="truncate font-medium leading-snug">
                        {chat.title || "Untitled Chat"}
                      </span>
                    </Link>
                  );
                })}
              </div>
            ))
          ) : (
            <div className="px-2 py-6 text-xs text-gray-400 italic text-center">
              ยังไม่มีประวัติการคุย
            </div>
          )}
        </div>

        {/* Bottom section */}
        <div className="pt-3 space-y-1.5 border-t mt-2" style={{ borderColor: "#e2eaf6" }}>
          {token ? (
            <>
              <div className="flex items-center gap-2 px-2 py-1.5 rounded-lg"
                style={{ background: "rgba(24,95,165,0.05)" }}>
                <div className="w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold text-white flex-shrink-0"
                  style={{ background: "linear-gradient(135deg, #8294C4, #6b7fbb)" }}>
                  {(username || "U")[0].toUpperCase()}
                </div>
                <span className="text-xs font-medium truncate" style={{ color: "#185fa5" }}>
                  {username || "User"}
                </span>
              </div>
              <button
                onClick={() => {
                  localStorage.removeItem("jp_token");
                  localStorage.removeItem("jp_username");
                  localStorage.removeItem("jp_user");
                  sessionStorage.removeItem("jp_screen");
                  sessionStorage.removeItem("jp_session_id");
                  window.location.reload();
                }}
                className="w-full flex items-center gap-2 px-2 py-1.5 rounded-lg text-xs transition hover:opacity-80"
                style={{ color: "#a32d2d", background: "rgba(163,45,45,0.05)" }}
              >
                <i className="bi bi-box-arrow-right text-xs"></i>
                <span className="font-medium">Logout</span>
              </button>
            </>
          ) : (
            <Link to="/login"
              className="w-full flex items-center justify-center gap-2 px-2 py-1.5 rounded-lg text-xs transition hover:opacity-80 no-underline"
              style={{ background: "rgba(24,95,165,0.08)", color: "#185fa5" }}>
              <span className="font-medium">Login</span>
            </Link>
          )}
        </div>
      </div>
    </>
  );
}

export default Sidebar;
