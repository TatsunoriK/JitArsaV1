import { HashRouter, Routes, Route, Navigate } from "react-router-dom";
import "bootstrap/dist/css/bootstrap.min.css";
import "./App.css";

// pages
import Chatbot from "../pages/Chatbot.jsx";
import Login from "../pages/LoginPage.jsx";
import Register from "../pages/Register.jsx";

function ProtectedRoute({ children }) {
  const token = localStorage.getItem("jp_token");

  if (!token) {
    return <Navigate to="/login" />;
  }

  return children;
}

function App() {
  return (
    <HashRouter>
      <Routes>
        <Route path="/" element={<Chatbot />} />
        <Route path="/register" element={<Register />} />
        <Route path="/login" element={<Login />} />

        {/* กันไม่ให้เข้า chatbot ถ้ายังไม่ login */}
        {/* <Route
          path="/"
          element={
            <ProtectedRoute>
              <Chatbot />
            </ProtectedRoute>
          }
        /> */}
      </Routes>
    </HashRouter>
  );
}

export default App;
