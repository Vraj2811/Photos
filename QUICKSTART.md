# ⚡ Quick Start Guide

Get up and running in 5 minutes!

## Step 1: Install Ollama

```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh
```

## Step 2: Start Ollama & Pull Models

```bash
# Terminal 1 - Start Ollama server
ollama serve
```

```bash
# Terminal 2 - Download required models (one-time setup)
ollama pull llava
ollama pull nomic-embed-text
```

## Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Step 4: Run the App

```bash
streamlit run streamlit_app.py
```

## Step 5: Use It!

1. **Upload Images**: Go to "📤 Upload Images" tab
2. **Add some images**: Click "Choose images to upload"
3. **Process**: Click "🚀 Process Images" (wait for AI descriptions)
4. **Search**: Go to "🔍 Search Images" tab
5. **Query**: Type something like "sunset" or "person smiling"
6. **Results**: See images ranked by confidence!

## 🎯 Test Search Examples

Try these queries after uploading diverse images:
- "landscape with mountains"
- "person smiling"
- "red car"
- "food on a plate"
- "technology and computers"

## ✅ Verify Setup

In the sidebar, you should see:
- ✅ llava (green checkmark)
- ✅ nomic-embed-text (green checkmark)
- System Ready

If you see ❌, check the troubleshooting section in README.md

## 🚨 Common First-Time Issues

### Issue: "Ollama connection failed"
**Solution:** Run `ollama serve` in a separate terminal

### Issue: "Model not found"
**Solution:** Run `ollama pull llava` and `ollama pull nomic-embed-text`

### Issue: Port already in use
**Solution:** Streamlit uses port 8501 by default. Change with:
```bash
streamlit run streamlit_app.py --server.port 8502
```

---

That's it! You're ready to search images with AI 🚀

