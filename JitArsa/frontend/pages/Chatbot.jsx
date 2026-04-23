import { useEffect, useRef, useState } from "react";

export default function Chatbot() {
  const [screen, setScreen] = useState("splash");
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

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
      const res = await fetch("/ask-pha", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // ✅ ส่ง history จริงๆ แทน []
        body: JSON.stringify({ question, history }),
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
            m.id === PLACEHOLDER_ID ? { ...m, text: snapshot } : m
          )
        );
      }

      setMessages((prev) =>
        prev.map((m) =>
          m.id === PLACEHOLDER_ID
            ? { ...m, id: `bot-${Date.now()}`, text: full || "ขอโทษนะ ภาหาคำตอบไม่เจอ" }
            : m
        )
      );
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

  if (screen === "splash") {
    return (
      <div
        className="min-h-screen flex flex-col items-center justify-center gap-5 text-center px-6 cursor-pointer bg-[#DBDFEA] "
        onClick={() => setScreen("chat")}
      >
        <div className="relative">
          <div className="absolute inset-0 rounded-full bg-[#ACB1D6] opacity-30 scale-125" />
          <img
            src="/girl.png"
            alt="team"
            className="relative w-28 h-28 rounded-full object-cover border-4 border-white shadow-lg"
          />
        </div>

        <div className="space-y-1">
          <h1 className="text-4xl font-bold tracking-tight text-[#8294C4]">
            Nop Napha
          </h1>
          <p className="text-[#8294C4] opacity-70 text-sm font-medium">
            เพื่อนที่ปรึกษาหางานอาสาของเธอ
          </p>
        </div>

        <p className="text-[#8294C4] text-base">มาเริ่มคุยกับภากันเถอะ !</p>

        <button
          className="mt-2 px-8 py-3 rounded-full bg-[#8294C4] text-white font-semibold text-sm shadow-md hover:bg-[#6b7db0] active:scale-95 transition-all duration-150"
          onClick={() => setScreen("chat")}
        >
          CHITCHAT
        </button>
      </div>
    );
  }

  const isEmpty = messages.length === 0;

  return (
    <div className="flex flex-col h-screen bg-[#DBDFEA] font-kodchasan">
      {/* Header */}
      <div className="sticky top-0 z-10 bg-white/80 backdrop-blur border-b border-[#ACB1D6]/30 px-4 py-3">
        <div className="flex items-center gap-3 max-w-3xl mx-auto">
          <img
            src="/girl.png"
            alt="bot"
            className="w-9 h-9 rounded-full object-cover border-2 border-[#ACB1D6]/40"
          />
          <div>
            <h2 className="text-sm font-bold text-[#8294C4] leading-tight">
              My friend name Pha
            </h2>
            <p className="text-xs text-[#ACB1D6]">เพื่อนคุยงานอาสา</p>
          </div>

          <div className="ml-auto flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
            <span className="text-xs text-[#ACB1D6]">ออนไลน์</span>
          </div>
        </div>
      </div>

      {isEmpty ? (
        <div className="flex-1 flex flex-col items-center justify-center gap-6 px-4 pb-28">
          <div className="w-full max-w-lg bg-white rounded-3xl p-8 text-center shadow-sm border border-[#DBDFEA]">
            <img
              src="/girl.png"
              alt="bot"
              className="w-20 h-20 rounded-full mx-auto mb-4 border-4 border-[#DBDFEA] shadow"
            />
            <p className="text-xl font-bold text-[#8294C4] mb-1">สวัสดี! 👋</p>
            <p className="text-[#8294C4] font-semibold mb-1">
              เราชื่อภา เป็นเพื่อนคุยงานอาสาของเธอ
            </p>
            <p className="text-sm text-[#ACB1D6] leading-relaxed">
              มีอะไรอยากถามเกี่ยวกับงานอาสาไหม?
              <br />
              ลองพิมพ์คำถามหรือเลือกจากคำแนะนำด้านล่างได้เลย!
            </p>

            <div className="flex flex-wrap justify-center gap-2 mt-5">
              {[
                "มีงานอาสาที่กรุงเทพไหม?",
                "อยากได้งานอาสาออนไลน์",
                "งานอาสาฟรีมีไหม?",
              ].map((s) => (
                <button
                  key={s}
                  onClick={() => setInput(s)}
                  className="px-3 py-1.5 rounded-full bg-[#DBDFEA] text-[#8294C4] text-xs font-medium hover:bg-[#ACB1D6] hover:text-white transition-colors"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>

          <div className="w-full max-w-2xl px-0">
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
          <div className="flex-1 overflow-y-auto px-4 py-5 pb-28 max-w-3xl w-full mx-auto space-y-4">
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex items-end gap-2 ${
                  msg.role === "user" ? "flex-row-reverse" : "flex-row"
                }`}
              >
                {msg.role === "bot" && (
                  <img
                    src="/girl.png"
                    alt="bot"
                    className="w-8 h-8 rounded-full object-cover flex-shrink-0 border-2 border-[#DBDFEA]"
                  />
                )}

                <div
                  className={`max-w-[75%] px-4 py-2.5 rounded-2xl text-sm leading-relaxed break-words ${
                    msg.role === "user"
                      ? "bg-[#8294C4] text-white rounded-br-sm shadow-sm"
                      : "bg-white text-[#374151] rounded-bl-sm shadow-sm border border-[#DBDFEA]"
                  }`}
                >
                  {msg.text.split("\n").map((line, i) => (
                    <p
                      key={i}
                      className={
                        i < msg.text.split("\n").length - 1 ? "mb-1" : ""
                      }
                    >
                      {line}
                    </p>
                  ))}
                </div>
              </div>
            ))}

            {loading && (
              <div className="flex items-end gap-2">
                <img
                  src="/girl.png"
                  alt="bot"
                  className="w-8 h-8 rounded-full object-cover flex-shrink-0 border-2 border-[#DBDFEA]"
                />
                <div className="bg-white border border-[#DBDFEA] rounded-2xl rounded-bl-sm px-4 py-3 shadow-sm flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-[#ACB1D6] animate-bounce [animation-delay:0ms]" />
                  <span className="w-2 h-2 rounded-full bg-[#ACB1D6] animate-bounce [animation-delay:150ms]" />
                  <span className="w-2 h-2 rounded-full bg-[#ACB1D6] animate-bounce [animation-delay:300ms]" />
                </div>
              </div>
            )}

            <div ref={bottomRef} />
          </div>

          <div className="fixed bottom-0 left-0 right-0 bg-[#DBDFEA]/90 backdrop-blur px-4 py-3 pb-5">
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
  );
}

function InputBar({ value, onChange, onSend, loading }) {
  return (
    <div className="flex items-center gap-2 bg-white rounded-2xl px-4 py-2 shadow-sm border border-[#ACB1D6]/30 focus-within:border-[#8294C4]/50 focus-within:shadow-md transition-all">
      <input
        value={value}
        placeholder="ถามภามาโลด !"
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && onSend()}
        disabled={loading}
        className="flex-1 bg-transparent outline-none text-sm text-[#374151] placeholder-[#ACB1D6] disabled:opacity-50 font-[inherit]"
      />
      <button
        onClick={onSend}
        disabled={loading}
        className="w-9 h-9 flex items-center justify-center rounded-xl bg-[#8294C4] text-white text-base font-bold shadow-sm hover:bg-[#6b7db0] active:scale-95 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
      >
        {loading ? (
          <span className="w-4 h-4 border-2 border-white/40 border-t-white rounded-full animate-spin" />
        ) : (
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="currentColor"
            className="w-4 h-4"
          >
            <path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" />
          </svg>
        )}
      </button>
    </div>
  );
}
