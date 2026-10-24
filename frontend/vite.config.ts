import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      // No rewrite: FastAPI mounts every router under /api (see backend/main.py)
      '/api': { target: 'http://localhost:8000' },
    },
  },
})
