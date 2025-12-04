# 🚀 React Web App Setup Guide

## Overview

This is a modern, responsive React + FastAPI web application for AI-powered image search.

**Stack:**
- **Frontend:** React 18 + TypeScript + Vite + Tailwind CSS
- **Backend:** FastAPI + Python 3.10+
- **AI:** Ollama (LLaVA vision model + nomic-embed-text)
- **Vector DB:** FAISS
- **Database:** SQLite + SQLAlchemy

---

## 📋 Prerequisites

1. **Node.js** (v18 or higher)
   ```bash
   node --version  # Should be v18+
   ```

2. **Python** (3.10 or higher)
   ```bash
   python3 --version  # Should be 3.10+
   ```

3. **Ollama** (already installed)
   ```bash
   ollama --version
   ```

4. **Required Models**
   ```bash
   ollama pull llava
   ollama pull nomic-embed-text
   ```

---

## 🛠️ Installation

### Step 1: Install Backend Dependencies

```bash
cd backend
pip3 install -r requirements.txt
```

### Step 2: Install Frontend Dependencies

```bash
cd ../frontend
npm install
```

This will install:
- React & React DOM
- TypeScript
- Vite (build tool)
- Tailwind CSS
- Axios (API client)
- Lucide React (icons)
- React Dropzone (file uploads)

---

## 🚀 Running the Application

You need to run **both** backend and frontend simultaneously.

### Terminal 1: Start Backend API

```bash
cd backend
python3 api.py
```

✅ Backend running at: **http://localhost:8000**

You should see:
```
🚀 Starting AI Image Search API...
📁 Images: /path/to/images
💾 Database: /path/to/image_search.db
🔢 Vectors: 147
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Terminal 2: Start Frontend Dev Server

```bash
cd frontend
npm run dev
```

✅ Frontend running at: **http://localhost:3000**

You should see:
```
  VITE v5.0.8  ready in 543 ms

  ➜  Local:   http://localhost:3000/
  ➜  Network: use --host to expose
  ➜  press h + enter to show help
```

---

## 🌐 Access the Application

Open your browser and navigate to:

### **http://localhost:3000**

You should see a beautiful gradient interface with three tabs:
- 🔍 **Search** - Natural language image search
- 📤 **Upload** - Drag & drop image upload
- 🖼️ **Gallery** - Browse all images

---

## 🎨 Features

### 1. **Search Tab**
- Natural language queries
- Real-time search
- Color-coded confidence badges
- Responsive grid layout

### 2. **Upload Tab**
- Drag & drop interface
- Multi-file upload
- Auto-processing with AI
- Real-time progress tracking
- Success/error feedback

### 3. **Gallery Tab**
- Browse all indexed images
- Infinite scroll (up to 100 images)
- Refresh button
- Responsive cards

### 4. **Status Bar**
- Total images count
- Total vectors count
- Ollama connection status
- Real-time updates every 10s

---

## 🔧 API Endpoints

The backend provides these REST APIs:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/status` | System status |
| POST | `/api/search` | Search images |
| GET | `/api/images` | Get all images |
| GET | `/api/images/{id}` | Get specific image |
| POST | `/api/upload` | Upload & process image |
| GET | `/api/vector/{id}` | Get vector stats |
| POST | `/api/rebuild-index` | Rebuild FAISS index |

### Example API Call (using curl):

```bash
# Search for images
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Nike shoes", "top_k": 5}'

# Get status
curl http://localhost:8000/api/status

# Get all images
curl http://localhost:8000/api/images?limit=10
```

---

## 🎯 How It Works

```
┌─────────────────────────────────────────────┐
│          React Frontend (Port 3000)         │
│  - Search interface                         │
│  - Upload interface                         │
│  - Gallery view                             │
└───────────────────┬─────────────────────────┘
                    │ REST API calls
                    │
┌───────────────────▼─────────────────────────┐
│        FastAPI Backend (Port 8000)          │
│  - Image processing                         │
│  - Ollama integration                       │
│  - Vector search                            │
│  - Database management                      │
└───────────────────┬─────────────────────────┘
                    │
        ┌───────────┴────────────┐
        │                        │
┌───────▼────────┐    ┌─────────▼────────┐
│   SQLite DB    │    │  FAISS Index     │
│  - Images      │    │  - Vectors       │
│  - Descriptions│    │  - Similarity    │
└────────────────┘    └──────────────────┘
```

---

## 🐛 Troubleshooting

### Frontend Issues

**Port 3000 already in use?**
```bash
# Kill process on port 3000
lsof -ti:3000 | xargs kill -9

# Or change port in vite.config.ts
server: {
  port: 3001  // Use different port
}
```

**Module not found errors?**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### Backend Issues

**Port 8000 already in use?**
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9
```

**Ollama not connected?**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start it
ollama serve
```

**CORS errors?**
The backend is configured to accept requests from:
- http://localhost:3000
- http://localhost:5173 (Vite default)

If you change the frontend port, update `backend/api.py`:
```python
allow_origins=["http://localhost:YOUR_PORT"]
```

### Search Not Working?

1. Check backend logs for errors
2. Ensure Ollama is running: `curl http://localhost:11434/api/tags`
3. Check if models are installed: `ollama list`
4. Rebuild index if needed: `POST /api/rebuild-index`

---

## 📦 Building for Production

### Build Frontend

```bash
cd frontend
npm run build
```

This creates an optimized build in `frontend/dist/`.

### Serve Production Build

```bash
# Using built-in preview
npm run preview

# Or use a static server
npx serve -s dist -p 3000
```

### Production Backend

```bash
cd backend
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 🎨 Customization

### Change Theme Colors

Edit `frontend/tailwind.config.js`:

```javascript
theme: {
  extend: {
    colors: {
      primary: '#your-color',
      secondary: '#your-color',
    }
  }
}
```

### Change Gradient Background

Edit `frontend/src/index.css`:

```css
body {
  background: linear-gradient(135deg, #your-color1 0%, #your-color2 100%);
}
```

### Add New Features

1. Add API endpoint in `backend/api.py`
2. Add API function in `frontend/src/api.ts`
3. Create component in `frontend/src/components/`
4. Import in `App.tsx`

---

## 🚀 Performance Tips

### Frontend
- Images are lazy-loaded
- API calls are debounced
- Status refreshes every 10s (not on every action)

### Backend
- FAISS index is cached in memory
- Database uses connection pooling
- Images served as static files (fast)

### Scaling
- Current setup: ~1000 images without issues
- For 10,000+ images:
  - Use PostgreSQL instead of SQLite
  - Consider Redis for caching
  - Use CDN for images
  - Add pagination

---

## 📊 Tech Stack Comparison

| Feature | Streamlit/Gradio | React + FastAPI |
|---------|------------------|-----------------|
| **Performance** | Good | Excellent ⚡ |
| **Customization** | Limited | Unlimited 🎨 |
| **Responsiveness** | Fair | Excellent 📱 |
| **Production Ready** | Prototypes | Yes ✅ |
| **Learning Curve** | Easy | Moderate |
| **API First** | No | Yes 🔌 |
| **Mobile Support** | Basic | Native 📲 |

---

## 🎉 What's New vs Streamlit/Gradio?

✅ **Modern UI** - Material design, glassmorphism, smooth animations  
✅ **Fully Responsive** - Works perfectly on mobile, tablet, desktop  
✅ **Real-time Updates** - WebSocket support ready  
✅ **Better Performance** - No page reloads, instant feedback  
✅ **API First** - Can be consumed by mobile apps, other services  
✅ **Production Ready** - Can deploy to Vercel, Netlify, AWS  
✅ **Customizable** - Full control over every pixel  
✅ **Scalable** - Can handle thousands of users  

---

## 📚 Resources

- **React Docs:** https://react.dev
- **FastAPI Docs:** https://fastapi.tiangolo.com
- **Vite Docs:** https://vitejs.dev
- **Tailwind CSS:** https://tailwindcss.com
- **Lucide Icons:** https://lucide.dev

---

## 🤝 Support

Having issues? Check:

1. Both servers are running (backend + frontend)
2. Ollama is running: `ollama serve`
3. Models are installed: `ollama list`
4. No port conflicts
5. Check browser console (F12) for errors
6. Check backend terminal for API errors

---

**Enjoy your new React web app! 🎉**





