import { useEffect, useRef, useState, useCallback } from "react";
import { useParams, useNavigate } from "react-router-dom";
import Sidebar from "../components/Sidebar";

export default function ChatHistory() {
  const { sessionId } = useParams();
  const navigate = useNavigate();
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [loadingMessages, setLoadingMessages] = useState(true);
  const bottomRef = useRef(null);
  const refreshHistoryRef = useRef(null);

  // โหลด messages เก่าจาก DB
  useEffect(() => {
    const fetchMessages = async () => {
      setLoadingMessages(true);
      try {
        const token = localStorage.getItem("jp_token");
        const res = await fetch(`/api/history/${sessionId}`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        const result = await res.json();
        if (result.success && result.data.length > 0) {
          // แปลง format จาก DB → format ที่ UI ใช้
          const mapped = result.data.map((m, i) => ({
            id: `db-${i}-${m._id}`,
            role: m.role === "assistant" ? "bot" : "user",
            text: m.content,
          }));
          setMessages(mapped);
        }
      } catch (err) {
        console.error("Error loading messages:", err);
      } finally {
        setLoadingMessages(false);
      }
    };
    fetchMessages();
    // set sessionId ใน sessionStorage เพื่อ highlight sidebar
    sessionStorage.setItem("jp_session_id", sessionId);
    sessionStorage.setItem("jp_screen", "chat");
  }, [sessionId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const onRegisterRefreshCallback = useCallback((fn) => {
    refreshHistoryRef.current = fn;
  }, []);

  const handleNewChat = useCallback(() => {
    const newId = crypto.randomUUID();
    sessionStorage.setItem("jp_session_id", newId);
    sessionStorage.setItem("jp_screen", "chat");
    navigate("/");
  }, [navigate]);

  async function sendMessage() {
    const question = input.trim();
    if (!question || loading) return;

    const userMsg = { id: `user-${Date.now()}`, role: "user", text: question };
    const updatedMessages = [...messages, userMsg];
    setMessages(updatedMessages);
    setInput("");
    setLoading(true);

    const history = updatedMessages
      .filter((m) => m.text && m.text.trim() !== "")
      .map((m) => ({ role: m.role === "bot" ? "assistant" : "user", content: m.text }));

    const PLACEHOLDER_ID = "loading-placeholder";
    setMessages((prev) => [...prev, { id: PLACEHOLDER_ID, role: "bot", text: "" }]);

    try {
      const token = localStorage.getItem("jp_token");
      const res = await fetch("/ask-pha", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({ question, history, sessionId }),
      });

      if (!res.ok) throw new Error(`Server error: ${res.status}`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let full = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        full += decoder.decode(value, { stream: true });
        const snapshot = full;
        setMessages((prev) =>
          prev.map((m) => m.id === PLACEHOLDER_ID ? { ...m, text: snapshot } : m)
        );
      }

      setMessages((prev) =>
        prev.map((m) =>
          m.id === PLACEHOLDER_ID
            ? { ...m, id: `bot-${Date.now()}`, text: full || "ขอโทษนะ ภาหาคำตอบไม่เจอ" }
            : m
        )
      );
      if (refreshHistoryRef.current) refreshHistoryRef.current();
    } catch (err) {
      console.error("Fetch Error:", err);
      setMessages((prev) => [
        ...prev.filter((m) => m.id !== PLACEHOLDER_ID),
        { id: `bot-error-${Date.now()}`, role: "bot", text: "เชื่อมต่อ backend ไม่ได้ ลองใหม่อีกทีนะ" },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex h-screen" style={{ fontFamily: "'Noto Sans Thai', sans-serif" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+Thai:wght@300;400;600;700&family=Noto+Sans+Thai:wght@300;400;500&display=swap');
        :root { --indigo: #8294C4; --lavender: #ACB1D6; --mist: #DBDFEA; }
      `}</style>

      <Sidebar onNewChat={handleNewChat} onRegisterRefresh={onRegisterRefreshCallback} />

      <div className="flex flex-col flex-1 relative" style={{ background: "var(--mist)" }}>
        {loadingMessages ? (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-sm text-gray-400">กำลังโหลดบทสนทนา...</div>
          </div>
        ) : (
          <>
            <div className="flex-1 overflow-y-auto px-4 py-5 pb-28 max-w-3xl w-full mx-auto space-y-4">
              {messages.map((msg) => (
                <div key={msg.id} className={`flex items-end gap-2 ${msg.role === "user" ? "flex-row-reverse" : "flex-row"}`}>
                  {msg.role === "bot" && (
                    <div className="w-10 h-10 rounded-full bg-white shadow-sm flex items-center justify-center flex-shrink-0">
                      <img src="/girl.png" alt="bot" className="w-full h-full rounded-full object-cover" />
                    </div>
                  )}
                  <div
                    className="max-w-[75%] mt-4 px-4 py-2.5 pt-3 rounded-2xl text-base leading-relaxed break-words"
                    style={
                      msg.role === "user"
                        ? { background: "linear-gradient(135deg, var(--indigo), #6b7fbb)", color: "#fff", borderBottomRightRadius: "4px", boxShadow: "0 4px 12px rgba(130,148,196,0.3)" }
                        : { background: "rgba(255,255,255,0.85)", color: "#374151", borderBottomLeftRadius: "4px", border: "1px solid rgba(172,177,214,0.35)", boxShadow: "0 2px 8px rgba(130,148,196,0.1)" }
                    }
                  >
                    {msg.text.split("\n").map((line, i, arr) => (
                      <p key={i} className={i < arr.length - 1 ? "mb-1" : ""}>{line}</p>
                    ))}
                  </div>
                </div>
              ))}

              {loading && (
                <div className="flex items-end gap-2">
                  <div className="w-10 h-10 rounded-full bg-white shadow-sm flex items-center justify-center">
                    <img src="/girl.png" alt="bot" className="w-full h-full rounded-full object-cover" />
                  </div>
                  <div className="rounded-2xl px-4 py-3 flex items-center gap-1.5"
                    style={{ background: "rgba(255,255,255,0.85)", border: "1px solid rgba(172,177,214,0.35)", borderBottomLeftRadius: "4px" }}>
                    <span className="w-2 h-2 rounded-full animate-bounce [animation-delay:0ms]" style={{ background: "var(--lavender)" }} />
                    <span className="w-2 h-2 rounded-full animate-bounce [animation-delay:150ms]" style={{ background: "var(--lavender)" }} />
                    <span className="w-2 h-2 rounded-full animate-bounce [animation-delay:300ms]" style={{ background: "var(--lavender)" }} />
                  </div>
                </div>
              )}
              <div ref={bottomRef} />
            </div>

            <div className="sticky bottom-0 left-0 right-0 px-4 py-2 pb-3"
              style={{ background: "rgba(219,223,234,0.85)", backdropFilter: "blur(12px)" }}>
              <div className="max-w-3xl mx-auto">
                <div className="flex items-center gap-2 px-4 py-2 transition-all"
                  style={{ background: "rgba(255,255,255,0.8)", backdropFilter: "blur(12px)", borderRadius: "16px", border: "1.5px solid rgba(172,177,214,0.4)", boxShadow: "0 2px 12px rgba(130,148,196,0.1)" }}>
                  <input
                    value={input}
                    placeholder="ถามภามาโลด !"
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && sendMessage()}
                    disabled={loading}
                    className="flex-1 bg-transparent outline-none text-sm disabled:opacity-50"
                    style={{ color: "#374151", fontFamily: "'Noto Sans Thai', sans-serif" }}
                  />
                  <button onClick={sendMessage} disabled={loading}
                    className="p-2 flex items-center justify-center rounded-full active:scale-90 disabled:opacity-40 transition-all duration-200">
                    {loading ? (
                      <span className="w-5 h-5 border-2 rounded-full animate-spin"
                        style={{ borderColor: "rgba(130,148,196,0.3)", borderTopColor: "var(--indigo)" }} />
                    ) : (
                      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5" style={{ color: "var(--indigo)" }}>
                        <path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" />
                      </svg>
                    )}
                  </button>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
