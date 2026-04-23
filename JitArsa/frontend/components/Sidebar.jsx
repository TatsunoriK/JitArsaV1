import { Link } from "react-router-dom";

function Sidebar() {
  const token = localStorage.getItem("jp_token");
  const username = localStorage.getItem("jp_username");

  return (
    <div
      className="w-73 h-screen flex flex-col p-4 border-r"
      style={{ background: "#fff", borderColor: "#c8d9f0" }}
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-4">
        <img
          src="/girl.png"
          alt="bot"
          className="w-9 h-9 rounded-full object-cover border-2 border-[#ACB1D6]/40"
        />
        <div className="leading-tight">
          <h5 className="text-sm font-medium mt-0" style={{ color: "#185fa5" }}>
            Nop Napha
          </h5>
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
          ลองพิมพ์คำถามหรือเลือกคำแนะนำได้เลย!
        </p>
      </div>

      {/* Nav */}
      <Link
        to="/"
        className="flex items-center gap-2.5 px-3 py-2.5 rounded-xl border text-sm font-medium"
        style={{
          background: "#e6f1fb",
          borderColor: "#b5d4f4",
          color: "#185fa5",
        }}
      >
        <span className="text-base">Chat</span>
      </Link>

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
              className="flex items-center gap-2.5 px-3 py-2.5 rounded-xl border"
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
              className="w-full flex items-center gap-2.5 px-3 py-2.5 rounded-xl border transition hover:opacity-80"
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
              className="w-full flex items-center justify-center gap-2.5 px-3 py-2.5 rounded-xl border transition hover:opacity-80"
              style={{
                background: "#fff",
                borderColor: "#c8d9f0",
                color: "#185fa5",
              }}
            >
              <i className="bi bi-box-arrow-in-right text-base"></i>
              <span className="text-sm font-medium">Login</span>
            </Link>
          </>
        )}
      </div>
    </div>
  );
}

export default Sidebar;
