import { HashRouter, Routes, Route } from "react-router-dom";
import 'bootstrap/dist/css/bootstrap.min.css';
import "./App.css";

// 
import Chatbot from "../pages/Chatbot.jsx"; 
import Login from "../pages/LoginPage.jsx"; 
import Register from "../pages/Register.jsx"; 

function App() {
  return (
    <HashRouter>
      <Routes>
        <Route path="/regis" element={<Register />} />
        <Route path="/login" element={<Login />} />
        <Route path="/" element={<Chatbot />} />
      </Routes>
    </HashRouter>
  );
}

export default App;