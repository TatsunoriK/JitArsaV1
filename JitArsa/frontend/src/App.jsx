import { HashRouter, Routes, Route, Navigate } from "react-router-dom";
import "bootstrap/dist/css/bootstrap.min.css";
import "./App.css";

import Chatbot from "../pages/Chatbot.jsx";
import Login from "../pages/LoginPage.jsx";
import Register from "../pages/Register.jsx";
import ChatHistory from "../pages/ChatHistory.jsx";

function ProtectedRoute({ children }) {
  const token = localStorage.getItem("jp_token");
  if (!token) return <Navigate to="/login" />;
  return children;
}

function App() {
  return (
    <HashRouter>
      <Routes>
        <Route path="/" element={<Chatbot />} />
        <Route path="/register" element={<Register />} />
        <Route path="/login" element={<Login />} />
        {/* ✅ route สำหรับดูประวัติ session เก่า */}
        <Route path="/chat/:sessionId" element={<ChatHistory />} />
      </Routes>
    </HashRouter>
  );
}

export default App;
