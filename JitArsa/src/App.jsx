import { useEffect, useRef, useState } from "react";
import "./App.css";

export default function App() {
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
      id: Date.now(),
      role: "user",
      text: question,
    };

    const firstTime = messages.length === 0;

    let updatedMessages;
    if (firstTime) {
      const botWelcome = {
        id: Date.now() - 1,
        role: "bot",
        text: "เริ่มต้นค้นหางานจิตอาสา",
      };
      updatedMessages = [botWelcome, userMsg];
      setMessages(updatedMessages);
    } else {
      updatedMessages = [...messages, userMsg];
      setMessages(updatedMessages);
    }

    setInput("");
    setLoading(true);

    // แปลง messages เป็น history format ที่ backend รับได้
    const history = messages
      .filter((m) => m.role === "user" || m.role === "bot")
      .map((m) => ({
        role: m.role === "bot" ? "assistant" : "user",
        content: m.text,
      }));

    try {
      const res = await fetch("http://127.0.0.1:8000/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question, history }),
      });

      const data = await res.json();

      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          role: "bot",
          text: data.answer || "ไม่สามารถตอบคำถามได้ในขณะนี้",
        },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          role: "bot",
          text: "เชื่อมต่อ backend ไม่ได้ กรุณาตรวจสอบว่า backend รันอยู่หรือไม่",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  if (screen === "splash") {
    return (
      <div className="splash" onClick={() => setScreen("chat")}>
        <img src="/team.png" alt="team" className="splash-logo" />
        <h1>JitArsa PhaPai</h1>
        <p>มาเริ่มต้นหางานอาสาด้วยกันเลย !</p>
        <button className="start-btn">เริ่มใช้งาน</button>
      </div>
    );
  }

  const isEmpty = messages.length === 0;

  return (
    <div className="chat">
      <div className="header">
        <div className="brand">
          <img src="/team.png" alt="bot" />
          <div>
            <h2>JitArsa PhaPai</h2>
            <p>ผู้ช่วยค้นหางานอาสา</p>
          </div>
        </div>
      </div>

      {isEmpty ? (
        <div className="empty-state">
          <div className="hero-card">
            <img src="/team.png" alt="bot" className="hero-logo" />
            <p className="greeting">เริ่มต้นค้นหางานจิตอาสา</p>
            <p className="sub-greeting">
              ลองพิมพ์ เช่น งานอาสากรุงเทพ, งานอาสาออนไลน์, งานอาสาฟรี
            </p>
          </div>

          <InputBar
            value={input}
            onChange={setInput}
            onSend={sendMessage}
            loading={loading}
          />
        </div>
      ) : (
        <>
          <div className="messages">
            {messages.map((msg) => (
              <div key={msg.id} className={`row ${msg.role}`}>
                {msg.role === "bot" && (
                  <img src="/team.png" alt="bot" className="avatar" />
                )}
                <div className={`bubble ${msg.role}`}>
                  {msg.text.split("\n").map((line, index) => (
                    <p key={index}>{line}</p>
                  ))}
                </div>
              </div>
            ))}

            {loading && (
              <div className="row bot">
                <img src="/team.png" alt="bot" className="avatar" />
                <div className="bubble bot loading-bubble">
                  <span className="dot"></span>
                  <span className="dot"></span>
                  <span className="dot"></span>
                </div>
              </div>
            )}

            <div ref={bottomRef} />
          </div>

          <div className="input-wrap">
            <InputBar
              value={input}
              onChange={setInput}
              onSend={sendMessage}
              loading={loading}
            />
          </div>
        </>
      )}
    </div>
  );
}

function InputBar({ value, onChange, onSend, loading }) {
  return (
    <div className="input-bar">
      <input
        value={value}
        placeholder="ถามได้เลย !"
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && onSend()}
        disabled={loading}
      />
      <button onClick={onSend} disabled={loading}>
        {loading ? "..." : "↑"}
      </button>
    </div>
  );
}