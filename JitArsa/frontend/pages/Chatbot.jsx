import { useEffect, useRef, useState, useCallback } from "react";
import Sidebar from "../components/Sidebar";

export default function Chatbot() {
  const [screen, setScreen] = useState(() => sessionStorage.getItem("jp_screen") || "splash");
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);
  const [mounted, setMounted] = useState(false);
  const usernameRef = useRef(null);
  const isEmpty = messages.length === 0;
  // ✅ persist sessionId ใน sessionStorage — F5 ยังคง session เดิม
  const sessionIdRef = useRef(
    sessionStorage.getItem("jp_session_id") || (() => {
      const id = crypto.randomUUID();
      sessionStorage.setItem("jp_session_id", id);
      return id;
    })()
  );
  const refreshHistoryRef = useRef(null); // ref เก็บ fetchHistory จาก Sidebar

  // ✅ New Chat — reset ข้อความ + สร้าง sessionId ใหม่
  const resetChat = useCallback(() => {
    setMessages([]);
    setInput("");
    setLoading(false);
    const newId = crypto.randomUUID();
    sessionIdRef.current = newId;
    sessionStorage.setItem("jp_session_id", newId); // ✅ เก็บ session ใหม่
    sessionStorage.setItem("jp_screen", "chat");
    setScreen("chat");
  }, []);

  // ✅ stable callback กัน re-render loop
  const onRegisterRefreshCallback = useCallback((fn) => {
    refreshHistoryRef.current = fn;
  }, []);

  useEffect(() => {
    setMounted(true);
    setTimeout(() => usernameRef.current?.focus(), 600);
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function sendMessage() {
    const question = input.trim();
    if (!question || loading) return;

    const userMsg = {
      id: `user-${Date.now()}-${Math.random()}`,
      role: "user",
      text: question,
    };

    let updatedMessages;
    if (messages.length === 0) {
      updatedMessages = [
        {
          id: `bot-greeting`,
          role: "bot",
          text: "สวัสดี ! วันนี้มีอะไรจะถามภาหรอ ?",
        },
        userMsg,
      ];
    } else {
      updatedMessages = [...messages, userMsg];
    }

    setMessages(updatedMessages);
    setInput("");
    setLoading(true);

    // ✅ แก้จุดนี้: build history จาก updatedMessages จริงๆ
    // กรองเฉพาะข้อความที่มีเนื้อหา และแปลง role "bot" → "assistant"
    const history = updatedMessages
      .filter((m) => m.text && m.text.trim() !== "")
      .map((m) => ({
        role: m.role === "bot" ? "assistant" : "user",
        content: m.text,
      }));

    const PLACEHOLDER_ID = "loading-placeholder";

    setMessages((prev) => [
      ...prev,
      { id: PLACEHOLDER_ID, role: "bot", text: "" },
    ]);

    try {
      const token = localStorage.getItem("jp_token");
      const res = await fetch("/ask-pha", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({ question, history, sessionId: sessionIdRef.current }),
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
          prev.map((m) =>
            m.id === PLACEHOLDER_ID ? { ...m, text: snapshot } : m,
          ),
        );
      }

      setMessages((prev) =>
        prev.map((m) =>
          m.id === PLACEHOLDER_ID
            ? {
                ...m,
                id: `bot-${Date.now()}`,
                text: full || "ขอโทษนะ ภาหาคำตอบไม่เจอ",
              }
            : m,
        ),
      );
      // ✅ refresh sidebar หลัง AI ตอบจบ
      if (refreshHistoryRef.current) refreshHistoryRef.current();
    } catch (err) {
      console.error("Fetch Error:", err);
      setMessages((prev) => [
        ...prev.filter((m) => m.id !== PLACEHOLDER_ID),
        {
          id: `bot-error-${Date.now()}`,
          role: "bot",
          text: "เชื่อมต่อ backend ไม่ได้ ลองใหม่อีกทีนะ",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+Thai:wght@300;400;600;700&family=Noto+Sans+Thai:wght@300;400;500&display=swap');

        :root {
          --indigo:   #8294C4;
          --lavender: #ACB1D6;
          --mist:     #DBDFEA;
          --peach:    #FFEAD2;
        }

        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        .jp-root {
          min-height: 100vh;
          display: flex;
          align-items: center;
          justify-content: center;
          font-family: 'Noto Sans Thai', sans-serif;
          background: var(--mist);
          position: relative;
          overflow: hidden;
        }

        /* Layered background */
        .jp-bg-blob {
          position: absolute;
          border-radius: 50%;
          filter: blur(80px);
          opacity: 0.55;
          pointer-events: none;
        }
        .jp-bg-blob-1 {
          width: 520px; height: 520px;
          background: var(--indigo);
          top: -120px; left: -140px;
        }
        .jp-bg-blob-2 {
          width: 380px; height: 380px;
          background: var(--peach);
          bottom: -80px; right: -100px;
        }
        .jp-bg-blob-3 {
          width: 260px; height: 260px;
          background: var(--lavender);
          top: 40%; left: 55%;
          opacity: 0.35;
        }

        /* Subtle grid overlay */
        .jp-grid {
          position: absolute;
          inset: 0;
          background-image:
            linear-gradient(rgba(130,148,196,0.06) 1px, transparent 1px),
            linear-gradient(90deg, rgba(130,148,196,0.06) 1px, transparent 1px);
          background-size: 40px 40px;
          pointer-events: none;
        }

        /* Card */
        .jp-card {
          position: relative;
          width: 420px;
          background: rgba(255,255,255,0.72);
          backdrop-filter: blur(24px);
          -webkit-backdrop-filter: blur(24px);
          border: 1px solid rgba(172,177,214,0.4);
          border-radius: 24px;
          padding: 48px 44px 44px;
          box-shadow:
            0 4px 6px rgba(130,148,196,0.06),
            0 20px 60px rgba(130,148,196,0.18),
            inset 0 1px 0 rgba(255,255,255,0.8);
          transform: translateY(${mounted ? "0" : "28px"});
          opacity: ${mounted ? "1" : "0"};
          transition: transform 0.7s cubic-bezier(.22,1,.36,1), opacity 0.7s ease;
        }

        @keyframes shake {
          0%,100% { transform: translateX(0); }
          20%      { transform: translateX(-7px); }
          40%      { transform: translateX(7px); }
          60%      { transform: translateX(-4px); }
          80%      { transform: translateX(4px); }
        }
        .jp-card.shaking { animation: shake 0.45s ease; }

        /* Logo / Brand */
        .jp-brand {
          text-align: center;
          margin-bottom: 36px;
        }
        .jp-logo-ring {
          width: 64px; height: 64px;
          margin: 0 auto 14px;
          border-radius: 18px;
          background: linear-gradient(135deg, var(--indigo), var(--lavender));
          display: flex; align-items: center; justify-content: center;
          box-shadow: 0 8px 24px rgba(130,148,196,0.35);
          position: relative;
          overflow: hidden;
        }
        .jp-logo-ring::after {
          content: '';
          position: absolute;
          inset: 0;
          background: linear-gradient(135deg, rgba(255,255,255,0.25), transparent);
          border-radius: inherit;
        }
        .jp-logo-icon {
          font-size: 28px;
          filter: drop-shadow(0 2px 4px rgba(0,0,0,0.2));
        }
        .jp-title {
          font-family: 'Noto Serif Thai', serif;
          font-size: 22px;
          font-weight: 700;
          color: #3a4270;
          letter-spacing: -0.3px;
          line-height: 1.3;
        }
        .jp-subtitle {
          font-size: 13px;
          color: var(--lavender);
          margin-top: 4px;
          font-weight: 400;
          letter-spacing: 0.5px;
        }

        /* Divider */
        .jp-divider {
          height: 1px;
          background: linear-gradient(90deg, transparent, var(--mist), transparent);
          margin-bottom: 32px;
        }

        /* Form */
        .jp-field {
          margin-bottom: 18px;
        }
        .jp-label {
          display: block;
          font-size: 12.5px;
          font-weight: 500;
          color: #6370a0;
          margin-bottom: 7px;
          letter-spacing: 0.3px;
        }
        .jp-input-wrap {
          position: relative;
        }
        .jp-input {
          width: 100%;
          padding: 13px 16px 13px 44px;
          border: 1.5px solid rgba(172,177,214,0.5);
          border-radius: 12px;
          font-family: 'Noto Sans Thai', sans-serif;
          font-size: 14.5px;
          color: #3a4270;
          background: rgba(219,223,234,0.25);
          outline: none;
          transition: border-color 0.2s, box-shadow 0.2s, background 0.2s;
        }
        .jp-input::placeholder { color: #b8bdd8; }
        .jp-input:focus {
          border-color: var(--indigo);
          background: rgba(255,255,255,0.85);
          box-shadow: 0 0 0 3.5px rgba(130,148,196,0.15);
        }
        .jp-icon {
          position: absolute;
          left: 14px;
          top: 50%;
          transform: translateY(-50%);
          font-size: 16px;
          pointer-events: none;
          opacity: 0.5;
        }
        .jp-eye {
          position: absolute;
          right: 13px;
          top: 50%;
          transform: translateY(-50%);
          background: none;
          border: none;
          cursor: pointer;
          padding: 4px;
          font-size: 16px;
          opacity: 0.45;
          transition: opacity 0.15s;
          line-height: 1;
        }
        .jp-eye:hover { opacity: 0.8; }

        /* Error */
        .jp-error {
          display: flex;
          align-items: center;
          gap: 7px;
          background: rgba(255,234,210,0.7);
          border: 1px solid rgba(255,160,100,0.3);
          border-radius: 10px;
          padding: 10px 14px;
          font-size: 13px;
          color: #b05b20;
          margin-bottom: 18px;
          animation: fadeIn 0.25s ease;
        }
        @keyframes fadeIn { from { opacity:0; transform:translateY(-4px); } to { opacity:1; transform:translateY(0); } }

        /* Button */
        .jp-btn {
          width: 100%;
          padding: 14px;
          border: none;
          border-radius: 13px;
          font-family: 'Noto Sans Thai', sans-serif;
          font-size: 15px;
          font-weight: 600;
          cursor: pointer;
          position: relative;
          overflow: hidden;
          background: linear-gradient(135deg, var(--indigo) 0%, #6b7fbb 100%);
          color: #fff;
          letter-spacing: 0.3px;
          box-shadow: 0 6px 20px rgba(130,148,196,0.38);
          transition: transform 0.15s, box-shadow 0.15s, opacity 0.15s;
          margin-top: 4px;
        }
        .jp-btn:hover:not(:disabled) {
          transform: translateY(-1px);
          box-shadow: 0 8px 28px rgba(130,148,196,0.48);
        }
        .jp-btn:active:not(:disabled) {
          transform: translateY(0);
          box-shadow: 0 4px 14px rgba(130,148,196,0.3);
        }
        .jp-btn:disabled { opacity: 0.7; cursor: not-allowed; }
        .jp-btn::after {
          content: '';
          position: absolute;
          inset: 0;
          background: linear-gradient(135deg, rgba(255,255,255,0.12), transparent);
          pointer-events: none;
        }

        /* Spinner */
        .jp-spinner {
          display: inline-block;
          width: 16px; height: 16px;
          border: 2.5px solid rgba(255,255,255,0.4);
          border-top-color: #fff;
          border-radius: 50%;
          animation: spin 0.7s linear infinite;
          vertical-align: middle;
          margin-right: 8px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        /* Footer note */
        .jp-footer {
          text-align: center;
          margin-top: 24px;
          font-size: 12px;
          color: #b0b6cc;
          line-height: 1.7;
        }

        @media (max-width: 480px) {
          .jp-card { width: 92vw; padding: 36px 28px 32px; }
        }
      `}</style>

      {/* Splash screen */}
      {screen === "splash" && (
        <div
          className="min-h-screen flex flex-col items-center justify-center gap-3 text-center px-6 cursor-pointer"
          style={{
            background: "var(--mist)",
            fontFamily: "'Noto Sans Thai', sans-serif",
          }}
          onClick={() => { setScreen("chat"); sessionStorage.setItem("jp_screen", "chat"); }}
        >
          <div className="relative">
            <img
              src="/IMG_5518.PNG"
              alt="team"
              className="relative w-55 h-55 rounded-full object-cover"
            />
          </div>

          <div className="space-y-1">
            <h1
              className="text-4xl font-bold tracking-tight"
              style={{
                fontFamily: "'Noto Serif Thai', serif",
                color: "var(--indigo)",
              }}
            >
              Nop Napha
            </h1>
            <p
              className="text-lg font-medium"
              style={{ color: "var(--indigo)", opacity: 0.7 }}
            >
              เพื่อนที่ปรึกษาหางานอาสาของเธอ
            </p>
          </div>

          <button
            className="p-5 px-8 py-3 rounded-full text-white text-sm shadow-md active:scale-95 transition-all duration-150"
            style={{
              borderRadius: "9999px",
              background: "linear-gradient(135deg, var(--indigo), #6b7fbb)",
              fontFamily: "'Noto Sans Thai', sans-serif",
            }}
            onClick={(e) => {
              e.stopPropagation();
              setScreen("chat"); sessionStorage.setItem("jp_screen", "chat");
            }}
          >
            มาเริ่มคุยกับภากันเถอะ !
          </button>
        </div>
      )}

      {/* Chat screen */}
      {screen !== "splash" && (
        <div
          className="flex h-screen"
          style={{ fontFamily: "'Noto Sans Thai', sans-serif" }}
        >
          <Sidebar onNewChat={resetChat} onRegisterRefresh={onRegisterRefreshCallback} />

          <div
            className="flex flex-col flex-1 relative"
            style={{ background: "var(--mist)" }}
          >
            {isEmpty ? (
              <div className="flex-1 flex flex-col items-center justify-center gap-6 px-4 pb-28">
                {/* Empty state card — jp-card style */}
                <div
                  className="w-full max-w-2xl text-center"
                  style={{
                    background: "rgba(255,255,255,0.72)",
                    backdropFilter: "blur(24px)",
                    WebkitBackdropFilter: "blur(24px)",
                    border: "1px solid rgba(172,177,214,0.4)",
                    borderRadius: "24px",
                    padding: "40px 36px",
                    boxShadow:
                      "0 4px 6px rgba(130,148,196,0.06), 0 20px 60px rgba(130,148,196,0.18), inset 0 1px 0 rgba(255,255,255,0.8)",
                  }}
                >
                  <img
                    src="/IMG_5529.PNG"
                    alt="bot"
                    className="w-30 h-30 rounded-full mx-auto mb-4 border-4 shadow"
                    style={{ borderColor: "var(--mist)" }}
                  />

                  {/* Suggestion chips */}
                  <div className="flex flex-wrap justify-center gap-2 mt-5">
                    {[
                      "มีงานอาสาที่กรุงเทพไหม?",
                      "อยากได้งานอาสาออนไลน์",
                      "งานอาสาฟรีมีไหม?",
                    ].map((s) => (
                      <button
                        key={s}
                        onClick={() => setInput(s)}
                        className="px-4 py-2 text-xs font-medium transition-all duration-150 active:scale-95"
                        style={{
                          borderRadius: "20px",
                          background: "rgba(219,223,234,0.6)",
                          color: "var(--indigo)",
                          border: "1px solid rgba(172,177,214,0.4)",
                          fontFamily: "'Noto Sans Thai', sans-serif",
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.background = "var(--lavender)";
                          e.currentTarget.style.color = "#fff";
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.background =
                            "rgba(219,223,234,0.6)";
                          e.currentTarget.style.color = "var(--indigo)";
                        }}
                      >
                        {s}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="w-full max-w-2xl">
                  <InputBar
                    value={input}
                    onChange={setInput}
                    onSend={sendMessage}
                    loading={loading}
                  />
                </div>
              </div>
            ) : (
              <>
                <div className="flex-1 overflow-y-auto px-4 py-5 pb-28 max-w-3xl w-full mx-auto space-y-4 custom-scrollbar">
                  {messages.map((msg) => (
                    <div
                      key={msg.id}
                      className={`flex items-end gap-2 ${msg.role === "user" ? "flex-row-reverse" : "flex-row"}`}
                    >
                      {msg.role === "bot" && (
                        <div className="w-10 h-10 rounded-full bg-white shadow-sm flex items-center justify-center">
                          <img
                            src="/IMG_5529.PNG"
                            alt="bot"
                            className="w-full h-full rounded-full object-cover"
                          />
                        </div>
                      )}
                      <div
                        className="max-w-[75%] mt-4 px-4 py-2.5 pt-3 rounded-2xl text-base leading-relaxed break-words"
                        style={
                          msg.role === "user"
                            ? {
                                background:
                                  "linear-gradient(135deg, var(--indigo), #6b7fbb)",
                                color: "#fff",
                                borderBottomRightRadius: "4px",
                                boxShadow: "0 4px 12px rgba(130,148,196,0.3)",
                              }
                            : {
                                background: "rgba(255,255,255,0.85)",
                                color: "#374151",
                                borderBottomLeftRadius: "4px",
                                border: "1px solid rgba(172,177,214,0.35)",
                                boxShadow: "0 2px 8px rgba(130,148,196,0.1)",
                              }
                        }
                      >
                        {msg.text.split("\n").map((line, i, arr) => (
                          <p
                            key={i}
                            className={i < arr.length - 1 ? "mb-1" : ""}
                          >
                            {line}
                          </p>
                        ))}
                      </div>
                    </div>
                  ))}

                  {loading && (
                    <div className="flex items-end gap-2">
                      <div className="w-10 h-10 rounded-full bg-white shadow-sm flex items-center justify-center">
                        <img
                          src="/IMG_5529.PNG"
                          alt="bot"
                          className="w-full h-full rounded-full object-cover "
                        />
                      </div>
                      <div
                        className="rounded-2xl px-4 py-3 flex items-center gap-1.5"
                        style={{
                          background: "rgba(255,255,255,0.85)",
                          border: "1px solid rgba(172,177,214,0.35)",
                          borderBottomLeftRadius: "4px",
                          boxShadow: "0 2px 8px rgba(130,148,196,0.1)",
                        }}
                      >
                        <span
                          className="w-2 h-2 rounded-full animate-bounce [animation-delay:0ms]"
                          style={{ background: "var(--lavender)" }}
                        />
                        <span
                          className="w-2 h-2 rounded-full animate-bounce [animation-delay:150ms]"
                          style={{ background: "var(--lavender)" }}
                        />
                        <span
                          className="w-2 h-2 rounded-full animate-bounce [animation-delay:300ms]"
                          style={{ background: "var(--lavender)" }}
                        />
                      </div>
                    </div>
                  )}

                  <div ref={bottomRef} />
                </div>

                <div
                  className="sticky bottom-0 left-0 right-0 px-4 py-2 pb-3"
                  style={{
                    background: "rgba(219,223,234,0.85)",
                    backdropFilter: "blur(12px)",
                  }}
                >
                  <div className="max-w-3xl mx-auto">
                    <InputBar
                      value={input}
                      onChange={setInput}
                      onSend={sendMessage}
                      loading={loading}
                    />
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </>
  );
}

function InputBar({ value, onChange, onSend, loading }) {
  return (
    <div
      className="flex items-center gap-2 px-4 py-2 transition-all"
      style={{
        background: "rgba(255,255,255,0.8)",
        backdropFilter: "blur(12px)",
        borderRadius: "16px",
        border: "1.5px solid rgba(172,177,214,0.4)",
        boxShadow: "0 2px 12px rgba(130,148,196,0.1)",
        fontFamily: "'Noto Sans Thai', sans-serif",
      }}
    >
      <input
        value={value}
        placeholder="ถามภามาโลด !"
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && onSend()}
        disabled={loading}
        className="flex-1 bg-transparent outline-none text-sm disabled:opacity-50"
        style={{ color: "#374151", fontFamily: "'Noto Sans Thai', sans-serif" }}
      />
      <button
        onClick={onSend}
        disabled={loading}
        className="p-2 flex items-center justify-center rounded-full active:scale-90 disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-200"
      >
        {loading ? (
          <span
            className="w-5 h-5 border-2 rounded-full animate-spin"
            style={{
              borderColor: "rgba(130,148,196,0.3)",
              borderTopColor: "var(--indigo)",
            }}
          />
        ) : (
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="currentColor"
            className="w-5 h-5"
            style={{ color: "var(--indigo)" }}
          >
            <path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" />
          </svg>
        )}
      </button>
    </div>
  );
}
