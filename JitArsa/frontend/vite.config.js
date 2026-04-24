import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/ask-pha": {
        target: "http://127.0.0.1:5000",
        changeOrigin: true,
      },
      "/api": {
        target: "http://127.0.0.1:5000",
        changeOrigin: true,
      },
      "/login": {
        target: "http://127.0.0.1:5000",
        changeOrigin: true,
      },
      "/register": {
        target: "http://127.0.0.1:5000",
        changeOrigin: true,
      },
    },
  },
});