import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://192.168.1.20:8000',
        changeOrigin: true,
      },
      '/images': {
        target: 'http://192.168.1.20:8000',
        changeOrigin: true,
      }
    }
  }
})





