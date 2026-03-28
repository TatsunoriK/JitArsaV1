import { useState, useRef, useEffect } from "react";
import "./App.css";

const VOLUNTEER_DATA = `ขนิดอาสาสมัคร ที่เปิดรับอยู่

1. อาสาช่วยเหลือสมัยด้วยนมน้ำ/ภาษามือ/Flash Card/ล่ามหูหนวก/ล่ามตาบอด/
อาสาช่วยอยู่วันที่ 7-8 มิ.ย. 2569

จาก มูลนิธิอาสาเพื่อสังคม

📅 วันรับสมัครสิ้นสุด : 3/6/2026
👥 จำนวนผู้รับสมัคร : 30
💰 ค่าใช้จ่าย : 100-200 บาท
📍 พื้นที่ปฏิบัติงาน : กรุงเทพมหานคร
🕐 ระยะเวลากิจกรรม : 09:00-17:30 น.
📧 ช่องทางการติดต่อ : Email volunteerservice@gmail.com
🔗 Link : volunteerpoint.org/งานอาสาสมัครรับสมัคร-23/60195/`;

export default function App() {
  const [screen, setScreen] = useState("splash");
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function sendMessage() {
    if (!input.trim()) return;

    // First message — bot greeting appears first
    if (messages.length === 0) {
      const botWelcome = { id: 0, role: "bot", text: "เริ่มต้นค้นหางานจิตอาสา" };
      const userMsg = { id: Date.now(), role: "user", text: input };
      setMessages([botWelcome, userMsg]);
      setInput("");
      setTimeout(() => {
        setMessages((prev) => [
          ...prev,
          { id: Date.now() + 1, role: "bot", text: VOLUNTEER_DATA },
        ]);
      }, 600);
      return;
    }

    const userMsg = { id: Date.now(), role: "user", text: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setTimeout(() => {
      setMessages((prev) => [
        ...prev,
        { id: Date.now() + 1, role: "bot", text: VOLUNTEER_DATA },
      ]);
    }, 600);
  }

  // ── Splash ──
  if (screen === "splash") {
    return (
      <div className="splash" onClick={() => setScreen("chat")}>
        <img src="/team.png" alt="team" />
        <h1>JitArsa PhaPai</h1>
        <p>มาเริ่มต้นหางานอาสาด้วยกันเลย !</p>
      </div>
    );
  }

  const isEmpty = messages.length === 0;

  // ── Chat ──
  return (
    <div className="chat">
      {/* Header */}
      <div className="header">
        <img src="/team.png" alt="bot" />
        <span>JitArsa PhaPai</span>
      </div>

      {/* Empty: centered greeting + input */}
      {isEmpty ? (
        <div className="empty-state">
          <p className="greeting">เริ่มต้นค้นหางานจิตอาสา</p>
          <InputBar value={input} onChange={setInput} onSend={sendMessage} />
        </div>
      ) : (
        <>
          {/* Messages */}
          <div className="messages">
            {messages.map((msg) => (
              <div key={msg.id} className={`row ${msg.role}`}>
                {msg.role === "bot" && (
                  <img src="/team.png" alt="bot" className="avatar" />
                )}
                <div className="bubble">{msg.text}</div>
              </div>
            ))}
            <div ref={bottomRef} />
          </div>

          {/* Input pinned bottom */}
          <div className="input-wrap">
            <InputBar value={input} onChange={setInput} onSend={sendMessage} />
          </div>
        </>
      )}
    </div>
  );
}

function InputBar({ value, onChange, onSend }) {
  return (
    <div className="input-bar">
      <input
        value={value}
        placeholder="ถามได้เลย !"
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && onSend()}
        autoFocus
      />
      <button onClick={onSend}>↑</button>
    </div>
  );
}
