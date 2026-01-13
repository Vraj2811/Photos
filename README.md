# 🔍 AI Image & Video Search System

A powerful media search system that uses AI to generate descriptions and find similar images/videos using vector similarity search.

## 🌟 Features

- **Multi-format Support**: Process both images and videos (local storage).
- **AI Descriptions**: Uses LLaVA to automatically generate detailed descriptions for images.
- **Vector Search**: Converts descriptions to embeddings using `nomic-embed-text` and stores them in FAISS for semantic search.
- **Face Detection & Grouping**: Automatically detects faces using `buffalo_l` and groups them across your collection.
- **Folders & Albums**: Organize your media into folders with physical directory synchronization.
- **Parallel Processing**: High-performance upload pipeline with parallel execution of AI tasks (saving, description, embedding, face detection, and thumbnail generation).
- **Modern UI**: Beautiful React-based frontend with glassmorphism, smooth animations, and real-time search.
- **Bulk Processing**: Dedicated script for indexing large existing collections.
- **Transactional Uploads**: Robust upload mechanism with automatic rollback on failure.


## 🔧 Architecture

```
Media Upload → Parallel Tasks:
               ├── Save File
               ├── LLaVA Description → Embedding Generation
               ├── Face Detection → Face Grouping
               └── Thumbnail Generation
                                ↓
                        [Database] + [FAISS Index]
                                ↓
User Query → Embedding → Vector Search → Ranked Results
```

## 📋 Prerequisites

1. **Python 3.11+**
2. **Node.js & npm** (for the frontend)
3. **Ollama** installed and running
4. **Required Ollama models:**
   - `llava` - for image descriptions
   - `nomic-embed-text` - for text embeddings

## 🚀 Installation & Setup

### 1. Install Ollama & Models

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull models
ollama pull llava
ollama pull nomic-embed-text
```

### 2. Backend Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start the backend
python3 backend/api.py
```

### 3. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

## 🎯 Usage

### Bulk Processing
To index a large folder of photos/videos:
```bash
python3 bulk_process.py /path/to/your/photos/ [--folder "FolderName"]
```
This will process all files in the directory and optionally group them into a specific folder in the system.

### Resetting the App
To clear all data and start fresh:
```bash
./RESET_APP.sh
```

## 📁 Project Structure

- `backend/`: FastAPI application, database logic, and AI processors.
- `frontend/`: React + Vite + Tailwind CSS frontend.
- `bulk_process.py`: Script for batch processing media files.
- `image_search.db`: SQLite database for metadata and faces.
- `faiss_indexes/`: FAISS vector storage.
- `uploads/`: Original media files storage.

## ⚙️ Configuration

Edit `backend/config.py` to change model names, storage paths, or database settings.

**Built with ❤️ using FastAPI, React, Ollama, and FAISS**
