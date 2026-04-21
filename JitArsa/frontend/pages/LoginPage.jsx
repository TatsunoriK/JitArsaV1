import { useState, useEffect, useRef } from "react";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:5000/api";

export default function LoginPage({ onLoginSuccess }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [shake, setShake] = useState(false);
  const [mounted, setMounted] = useState(false);
  const usernameRef = useRef(null);

  useEffect(() => {
    setMounted(true);
    setTimeout(() => usernameRef.current?.focus(), 600);
  }, []);

  const triggerShake = () => {
    setShake(true);
    setTimeout(() => setShake(false), 500);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!username.trim() || !password) {
      setError("กรุณากรอกชื่อผู้ใช้และรหัสผ่าน");
      triggerShake();
      return;
    }

    setLoading(true);
    setError("");

    try {
      const res = await fetch(`${API_BASE}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: username.trim(), password }),
      });

      const data = await res.json();

      if (data.success) {
        localStorage.setItem("jp_token", data.token);
        localStorage.setItem("jp_user", JSON.stringify(data.user));
        onLoginSuccess?.(data);
      } else {
        setError(data.message || "เกิดข้อผิดพลาด");
        triggerShake();
      }
    } catch {
      setError("ไม่สามารถเชื่อมต่อเซิร์ฟเวอร์ได้");
      triggerShake();
    } finally {
      setLoading(false);
    }
  };

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

      <div className="jp-root">
        <div className="jp-bg-blob jp-bg-blob-1" />
        <div className="jp-bg-blob jp-bg-blob-2" />
        <div className="jp-bg-blob jp-bg-blob-3" />
        <div className="jp-grid" />

        <div className={`jp-card${shake ? " shaking" : ""}`}>
          {/* Brand */}
          <div className="jp-brand">
            <div className="jp-logo-ring">
              <span className="jp-logo-icon">🌸</span>
            </div>
            <div className="jp-title">จิตอาสา ผาไผ่</div>
            <div className="jp-subtitle">AI Chatbot · ระบบจัดการ</div>
          </div>

          <div className="jp-divider" />

          <form onSubmit={handleSubmit} autoComplete="off" noValidate>
            {/* Username */}
            <div className="jp-field">
              <label className="jp-label" htmlFor="jp-username">ชื่อผู้ใช้</label>
              <div className="jp-input-wrap">
                <span className="jp-icon">👤</span>
                <input
                  id="jp-username"
                  ref={usernameRef}
                  className="jp-input"
                  type="text"
                  placeholder="กรอกชื่อผู้ใช้"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  autoComplete="username"
                  spellCheck={false}
                />
              </div>
            </div>

            {/* Password */}
            <div className="jp-field">
              <label className="jp-label" htmlFor="jp-password">รหัสผ่าน</label>
              <div className="jp-input-wrap">
                <span className="jp-icon">🔒</span>
                <input
                  id="jp-password"
                  className="jp-input"
                  type={showPassword ? "text" : "password"}
                  placeholder="กรอกรหัสผ่าน"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  autoComplete="current-password"
                  style={{ paddingRight: "44px" }}
                />
                <button
                  type="button"
                  className="jp-eye"
                  onClick={() => setShowPassword((v) => !v)}
                  tabIndex={-1}
                  aria-label={showPassword ? "ซ่อนรหัสผ่าน" : "แสดงรหัสผ่าน"}
                >
                  {showPassword ? "🙈" : "👁️"}
                </button>
              </div>
            </div>

            {/* Error */}
            {error && (
              <div className="jp-error" role="alert">
                <span>⚠️</span>
                <span>{error}</span>
              </div>
            )}

            {/* Submit */}
            <button className="jp-btn" type="submit" disabled={loading}>
              {loading ? (
                <>
                  <span className="jp-spinner" />
                  กำลังเข้าสู่ระบบ…
                </>
              ) : (
                "เข้าสู่ระบบ"
              )}
            </button>
          </form>

          <div className="jp-footer">
            ระบบสำหรับเจ้าหน้าที่เท่านั้น<br />
            หากพบปัญหา กรุณาติดต่อผู้ดูแลระบบ
          </div>
        </div>
      </div>
    </>
  );
}