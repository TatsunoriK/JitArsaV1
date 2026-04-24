import { Link, useNavigate } from "react-router-dom";
import { useState, useEffect, useRef } from "react";

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
  const [confirmDeleteId, setConfirmDeleteId] = useState(null); // id ที่รอ confirm
  const [deletingId, setDeletingId] = useState(null);
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
      if (result.success) setChatHistory(result.data);
    } catch (error) {
      console.error("Error fetching history:", error);
    } finally {
      setLoadingHistory(false);
    }
  };

  useEffect(() => { fetchHistoryRef.current(); }, []);
  useEffect(() => {
    if (onRegisterRefresh) onRegisterRefresh(() => fetchHistoryRef.current());
  }, [onRegisterRefresh]);

  const handleDelete = async (e, sessionId) => {
    e.preventDefault();
    e.stopPropagation();
    setConfirmDeleteId(sessionId);
  };

  const confirmDelete = async (e, sessionId) => {
    e.preventDefault();
    e.stopPropagation();
    setDeletingId(sessionId);
    try {
      const t = localStorage.getItem("jp_token");
      await fetch(`/api/history/${sessionId}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${t}` },
      });
      // ถ้ากำลังดู session นี้อยู่ → กลับหน้าหลัก
      const currentSession = sessionStorage.getItem("jp_session_id");
      if (currentSession === sessionId) {
        const newId = crypto.randomUUID();
        sessionStorage.setItem("jp_session_id", newId);
        navigate("/");
      }
      setChatHistory((prev) => prev.filter((c) => c._id !== sessionId));
    } catch (err) {
      console.error("Delete error:", err);
    } finally {
      setDeletingId(null);
      setConfirmDeleteId(null);
    }
  };

  const cancelDelete = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setConfirmDeleteId(null);
  };

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
      <style>{`
        .custom-sidebar::-webkit-scrollbar { width: 4px; }
        .custom-sidebar::-webkit-scrollbar-thumb { background: rgba(172,177,214,0.4); border-radius: 10px; }
        .chat-item { position: relative; }
        .chat-item .delete-btn { opacity: 0; transition: opacity 0.15s; }
        .chat-item:hover .delete-btn { opacity: 1; }
      `}</style>

      <div className="w-56 h-screen flex flex-col px-3 py-4 border-r"
        style={{ background: "rgba(255,255,255,0.6)", backdropFilter: "blur(16px)", WebkitBackdropFilter: "blur(16px)", borderRight: "1px solid rgba(172,177,214,0.4)" }}>

        {/* Header */}
        <div className="flex items-center gap-2 mb-3">
          <img src="/girl.png" alt="bot" className="w-8 h-8 rounded-full object-cover border-2 border-[#ACB1D6]/40" />
          <h4 className="text-sm font-semibold" style={{ color: "#185fa5" }}>Nop Napha</h4>
        </div>

        {/* New Chat */}
        <button onClick={handleNewChat}
          className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-xl text-sm font-medium mb-3 transition-all hover:opacity-80 active:scale-95"
          style={{ background: "linear-gradient(135deg, #8294C4, #6b7fbb)", color: "#fff", border: "none", boxShadow: "0 4px 12px rgba(130,148,196,0.3)" }}>
          ✏️ New Chat
        </button>

        {/* History list */}
        <div className="flex-1 overflow-y-auto pr-1 custom-sidebar">
          {loadingHistory ? (
            <div className="px-2 py-4 text-xs text-gray-400 text-center">กำลังโหลด...</div>
          ) : grouped.length > 0 ? (
            grouped.map(([label, chats]) => (
              <div key={label} className="mb-3">
                <p className="text-[10px] font-semibold px-2 mb-1 uppercase tracking-widest" style={{ color: "#9ca3af" }}>
                  {label}
                </p>
                {chats.map((chat) => {
                  const isActive = chat._id === currentSessionId;
                  const isConfirming = confirmDeleteId === chat._id;
                  const isDeleting = deletingId === chat._id;

                  return (
                    <div key={chat._id} className="chat-item mb-0.5">
                      {isConfirming ? (
                        // Confirm delete UI
                        <div className="px-2 py-2 rounded-lg text-xs"
                          style={{ background: "rgba(163,45,45,0.06)", border: "1px solid rgba(163,45,45,0.2)" }}>
                          <p className="text-[11px] font-medium mb-1.5 truncate" style={{ color: "#374151" }}>
                            ลบ "{chat.title || "Untitled"}"?
                          </p>
                          <div className="flex gap-1.5">
                            <button onClick={(e) => confirmDelete(e, chat._id)}
                              className="flex-1 py-1 rounded-md text-[11px] font-semibold transition hover:opacity-80"
                              style={{ background: "#a32d2d", color: "#fff" }}>
                              {isDeleting ? "กำลังลบ..." : "ลบ"}
                            </button>
                            <button onClick={cancelDelete}
                              className="flex-1 py-1 rounded-md text-[11px] font-medium transition hover:opacity-80"
                              style={{ background: "rgba(0,0,0,0.06)", color: "#374151" }}>
                              ยกเลิก
                            </button>
                          </div>
                        </div>
                      ) : (
                        <Link to={`/chat/${chat._id}`}
                          className="flex items-center gap-2 px-2 py-2 rounded-lg text-xs transition-all no-underline hover:no-underline group"
                          style={{ color: isActive ? "#185fa5" : "#4a5568", background: isActive ? "rgba(24,95,165,0.1)" : "transparent" }}
                          onMouseEnter={(e) => { if (!isActive) e.currentTarget.style.background = "rgba(24,95,165,0.06)"; }}
                          onMouseLeave={(e) => { if (!isActive) e.currentTarget.style.background = "transparent"; }}>
                          <i className={`bi bi-chat-text text-xs flex-shrink-0 ${isActive ? "text-[#185fa5]" : "opacity-30 group-hover:opacity-60"}`}></i>
                          <span className="truncate font-medium leading-snug flex-1">{chat.title || "Untitled Chat"}</span>
                          {/* ปุ่มลบ — ขึ้นเมื่อ hover */}
                          <button
                            onClick={(e) => handleDelete(e, chat._id)}
                            className="delete-btn flex-shrink-0 w-5 h-5 flex items-center justify-center rounded hover:bg-red-100 transition-all"
                            title="ลบการสนทนานี้"
                            style={{ color: "#a32d2d" }}>
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-3.5 h-3.5">
                              <path fillRule="evenodd" d="M8.75 1A2.75 2.75 0 006 3.75v.443c-.795.077-1.584.176-2.365.298a.75.75 0 10.23 1.482l.149-.022.841 10.518A2.75 2.75 0 007.596 19h4.807a2.75 2.75 0 002.742-2.53l.841-10.52.149.023a.75.75 0 00.23-1.482A41.03 41.03 0 0014 4.193V3.75A2.75 2.75 0 0011.25 1h-2.5zM10 4c.84 0 1.673.025 2.5.075V3.75c0-.69-.56-1.25-1.25-1.25h-2.5c-.69 0-1.25.56-1.25 1.25v.325C8.327 4.025 9.16 4 10 4zM8.58 7.72a.75.75 0 00-1.5.06l.3 7.5a.75.75 0 101.5-.06l-.3-7.5zm4.34.06a.75.75 0 10-1.5-.06l-.3 7.5a.75.75 0 101.5.06l.3-7.5z" clipRule="evenodd" />
                            </svg>
                          </button>
                        </Link>
                      )}
                    </div>
                  );
                })}
              </div>
            ))
          ) : (
            <div className="px-2 py-6 text-xs text-gray-400 italic text-center">ยังไม่มีประวัติการคุย</div>
          )}
        </div>

        <div className="flex-1" />

        {/* Bottom */}
        <div className="pt-3 space-y-1.5 border-t mt-2" style={{ borderColor: "#e2eaf6" }}>
          {token ? (
            <>
              <div className="flex items-center gap-2 px-2 py-1.5 rounded-lg" style={{ background: "rgba(24,95,165,0.05)" }}>
                <div className="w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold text-white flex-shrink-0"
                  style={{ background: "linear-gradient(135deg, #8294C4, #6b7fbb)" }}>
                  {(username || "U")[0].toUpperCase()}
                </div>
                <span className="text-xs font-medium truncate" style={{ color: "#185fa5" }}>{username || "User"}</span>
              </div>
              <button onClick={() => {
                localStorage.removeItem("jp_token");
                localStorage.removeItem("jp_username");
                localStorage.removeItem("jp_user");
                sessionStorage.removeItem("jp_screen");
                sessionStorage.removeItem("jp_session_id");
                window.location.reload();
              }}
                className="w-full flex items-center gap-2 px-2 py-1.5 rounded-lg text-xs transition hover:opacity-80"
                style={{ color: "#a32d2d", background: "rgba(163,45,45,0.05)" }}>
                <i className="bi bi-box-arrow-right text-xs"></i>
                <span className="font-medium">Logout</span>
              </button>
            </>
          ) : (
            <Link to="/login" className="w-full flex items-center justify-center gap-2 px-2 py-1.5 rounded-lg text-xs transition hover:opacity-80 no-underline"
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
