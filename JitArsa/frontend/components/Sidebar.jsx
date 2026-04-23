import { Link } from "react-router-dom";
import { useState, useEffect } from "react";

function Sidebar() {
  const token = localStorage.getItem("jp_token");
  const username = localStorage.getItem("jp_username");
  const [chatHistory, setChatHistory] = useState([]);

  useEffect(() => {
    if (token) {
      fetchHistory();
    }
  }, [token]);

  const fetchHistory = async () => {
    try {
      const response = await fetch("http://localhost:5000/api/history");
      const result = await response.json();
      if (result.success) {
        setChatHistory(result.data);
      }

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const data = await response.json();
      setChatHistory(data);
    } catch (error) {
      console.error("Error fetching history:", error);
    }
  };

  return (
    <>
      <style>
        {`
        .custom-sidebar::-webkit-scrollbar {
        width: 4px;
        }
        
        .custom-sidebar::-webkit-scrollbar-thumb {
        background: rgba(172, 177, 214, 0.4);
        border-radius: 10px;
        }`}
      </style>

      <div
        className="w-72 h-screen flex flex-col px-4 py-4 border-r"
        style={{
          background: "rgba(255,255,255,0.6)",
          backdropFilter: "blur(16px)",
          WebkitBackdropFilter: "blur(16px)",
          borderRight: "1px solid rgba(172,177,214,0.4)",
        }}
      >
        {/* Header */}
        <div className="flex items-center gap-2 mb-4">
          <img
            src="/girl.png"
            alt="bot"
            className="w-9 h-9 rounded-full object-cover border-2 border-[#ACB1D6]/40"
          />
          <div className="leading-tight">
            <h4
              className="text-sm font-medium mt-0"
              style={{ color: "#185fa5" }}
            >
              Nop Napha
            </h4>
          </div>
        </div>

        {/* Greeting card */}
        <div className="rounded-xl mb-2">
          <p className="text-2xl font-medium mb-1" style={{ color: "#185fa5" }}>
            สวัสดี! 👋
          </p>
          <p className="text-lg font-medium mb-2" style={{ color: "#378add" }}>
            เราชื่อภา
            <br />
            เป็นเพื่อนคุยงานอาสาของเธอ
          </p>
          <p className="text-base leading-relaxed" style={{ color: "#5f5e5a" }}>
            มีอะไรอยากถามเกี่ยวกับงานอาสาไหม?
            <br />
            พิมพ์คำถามหรือเลือกคำแนะนำได้เลย!
          </p>
        </div>

        {/* ส่วนแสดงรายการประวัติแชท */}
        <div className="flex-1 overflow-y-auto mt-4 space-y-1 pr-1 custom-sidebar">
          <p className="text-[11px] font-bold px-3 mb-2 text-gray-400 uppercase tracking-widest">
            Recent Chats
          </p>

          {chatHistory.length > 0 ? (
            chatHistory.map((chat) => (
              <Link
                key={chat._id}
                to={`/chat/${chat._id}`}
                className="flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm transition-all no-underline hover:no-underline group"
                style={{ color: "#4a5568" }}
                onMouseEnter={(e) =>
                  (e.currentTarget.style.backgroundColor =
                    "rgba(24, 95, 165, 0.08)")
                }
                onMouseLeave={(e) =>
                  (e.currentTarget.style.backgroundColor = "transparent")
                }
              >
                <i className="bi bi-chat-right-text text-xs opacity-40 group-hover:text-[#185fa5] group-hover:opacity-100"></i>
                <span className="truncate flex-1 group-hover:text-[#185fa5] font-medium">
                  {chat.title || "Untitled Chat"}
                </span>
              </Link>
            ))
          ) : (
            <div className="px-3 py-4 text-xs text-gray-400 italic text-center border border-dashed rounded-xl border-gray-200">
              ยังไม่มีประวัติการคุย
            </div>
          )}
        </div>

        {/* Nav */}
        {/* <Link
          to="/"
          className="flex items-center gap-2.5 px-3 py-2.5 rounded-xl border text-sm font-medium"
          style={{
            background: "#e6f1fb",
            borderColor: "#b5d4f4",
            color: "#185fa5",
          }}
        >
          <span className="text-base">Chat</span>
        </Link> */}

        <div className="flex-1" />

        {/* Bottom section */}
        <div
          className="pt-4 space-y-2 border-t"
          style={{ borderColor: "#c8d9f0" }}
        >
          {token ? (
            <>
              {/* User Card */}
              <div
                className="flex items-center gap-2.5 px-3 py-2 mb-2 rounded-xl border"
                style={{ background: "#fff", borderColor: "#c8d9f0" }}
              >
                <i className="bi bi-person-fill text-[#185fa5] text-base"></i>

                <span
                  className="text-sm font-medium truncate"
                  style={{ color: "#185fa5" }}
                >
                  {username || "User"}
                </span>
              </div>

              {/* Logout Button */}
              <button
                onClick={() => {
                  localStorage.removeItem("jp_token");
                  localStorage.removeItem("jp_username");
                  window.location.reload();
                }}
                className="w-full flex items-center gap-2.5 px-3 py-2 rounded-xl border transition hover:opacity-80"
                style={{
                  color: "#a32d2d",
                  borderColor: "#f0c1c1",
                  background: "#fff",
                  borderRadius: "12px",
                }}
              >
                <i className="bi bi-box-arrow-right text-base"></i>

                <span className="text-sm font-medium">Logout</span>
              </button>
            </>
          ) : (
            <>
              {/* Login Button */}
              <Link
                to="/login"
                className="w-full flex items-center justify-center gap-2.5 px-3 py-2 rounded-xl border transition hover:opacity-80 no-underline hover:no-underline"
                style={{
                  background: "#fff",
                  borderColor: "#c8d9f0",
                  color: "#185fa5",
                }}
              >
                <span className="text-sm font-medium">Login</span>
              </Link>
            </>
          )}
        </div>
      </div>
    </>
  );
}

export default Sidebar;
