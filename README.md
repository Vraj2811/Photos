# 🔍 AI Image Search System

A powerful image search system that uses AI to generate descriptions and find similar images using vector similarity search.

## 🌟 Features

- **Upload Images**: Drag & drop through modern UI
- **AI Descriptions**: LLaVA automatically generates 2-3 line descriptions
- **Vector Search**: Converts descriptions to embeddings and stores in FAISS
- **Auto-Search**: Real-time search as you type (no button needed!)
- **Confidence Badges**: Color-coded match quality indicators
- **Modern UI**: Beautiful gradient design with smooth animations
- **Smart Gallery**: Filter and sort your image collection
- **Debug Mode**: Built-in troubleshooting tools

## 🔧 Architecture

```
Image Upload → LLaVA Description → Embedding Generation → Storage
                                          ↓
                                    [Database] + [FAISS Index]
                                          ↓
User Query → Embedding → Vector Search → Ranked Results
```

## 📋 Prerequisites

1. **Python 3.8+**
2. **Ollama** installed and running
3. **Required Ollama models:**
   - `llava` - for image descriptions
   - `nomic-embed-text` - for text embeddings

## 🚀 Installation

### 1. Install Ollama

```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama
ollama serve
```

### 2. Pull Required Models

```bash
# In a new terminal
ollama pull llava
ollama pull nomic-embed-text
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## 🎯 Usage

### Start the Application

**You have TWO UI options!**

#### Option 1: Streamlit (Modern gradient UI)
```bash
streamlit run streamlit_app.py
```
Opens at `http://localhost:8501`

#### Option 2: Gradio (Simple, clean UI)
```bash
python gradio_app.py
```
Opens at `http://localhost:7860`

**Both use the same database and work identically!** See [GRADIO_VS_STREAMLIT.md](GRADIO_VS_STREAMLIT.md) for comparison.

### Using the System

#### **Method 1: Bulk Processing (Recommended for many images)**

1. Copy images to the `images/` folder
2. Run the bulk processing script:
```bash
python build_vectors.py
```
3. Start the app and search!

See [BULK_PROCESS.md](BULK_PROCESS.md) for detailed guide.

#### **Method 2: Upload Through UI (Best for occasional uploads)**

**Search Images (Tab 1)**
1. Enter a search query (e.g., "sunset over mountains")
2. Results appear automatically as you type
3. View results ranked by confidence

**Upload Images (Tab 2)**
1. Click "Choose images to upload"
2. Select one or more images
3. Click "Process Images"
4. LLaVA generates descriptions automatically
5. Embeddings stored in FAISS for searching

**Browse All (Tab 3)**
- View all uploaded images in gallery
- Filter and sort your collection
- See their AI-generated descriptions

### Sidebar Actions

- **🔨 Rebuild Vector Index**: Rebuilds FAISS index from database
  - Use this if index gets out of sync
  - Or if you manually added images to the database
- **🔄 Refresh**: Refresh system status

## 📁 Project Structure

```
Image_Search/
├── streamlit_app.py          # Main application
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── images/                    # Uploaded images stored here
│   └── (uploaded images)
├── faiss_indexes/             # Vector index storage
│   ├── vectors.index          # FAISS index file
│   └── mapping.json           # Index to DB ID mapping
└── image_search.db            # SQLite database
```

## 🗄️ Database Schema

```sql
images (
    id INTEGER PRIMARY KEY,
    filename VARCHAR(255),
    file_path TEXT,
    description TEXT,          -- AI-generated description
    embedding TEXT,            -- JSON array of embedding vector
    created_at DATETIME
)
```

## 🔍 How Vector Search Works

1. **Indexing:**
   - Image uploaded → LLaVA generates description
   - Description converted to 768-dimensional embedding
   - Embedding normalized for cosine similarity
   - Stored in both SQLite (JSON) and FAISS (binary)

2. **Searching:**
   - User query → Converted to embedding
   - FAISS finds K nearest neighbors (cosine similarity)
   - Results mapped back to database records
   - Sorted by confidence score (0-1)

## ⚙️ Configuration

Edit the `Config` class in `streamlit_app.py`:

```python
class Config:
    VISION_MODEL = "llava"              # Ollama vision model
    EMBEDDING_MODEL = "nomic-embed-text" # Ollama embedding model
    EMBEDDING_DIMENSION = 768            # Embedding vector size
```

## 🐛 Troubleshooting

### Search Not Working?
1. **Check sidebar status** - Make sure Images = Vectors
2. **Click "🔨 Rebuild Vector Index"** if they don't match
3. **Verify Ollama is running**: `ollama serve`
4. **Check models are installed**: `ollama list`

For detailed troubleshooting, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

### "Ollama connection failed"
```bash
# Terminal 1
ollama serve

# Terminal 2
ollama pull llava
ollama pull nomic-embed-text
```

### "No results found" but images exist
- Click "🔨 Rebuild Vector Index" in sidebar
- Wait for "✅ Rebuilt X vectors!" message
- Try search again

### Quick Reset
```bash
rm -rf faiss_indexes/*
streamlit run streamlit_app.py
# Then click Rebuild Index in sidebar
```

## 🎨 Customization

### Change Description Length
Edit the prompt in `generate_description()`:
```python
'content': 'Describe this image in 2-3 clear, concise sentences...'
```

### Adjust Search Results
Change default number of results in search tab:
```python
top_k = st.selectbox("Results", [5, 10, 15, 20], index=0)
```

### Different Embedding Model
Try other Ollama embedding models:
```python
EMBEDDING_MODEL = "all-minilm"  # Smaller, faster
EMBEDDING_MODEL = "mxbai-embed-large"  # Larger, more accurate
```

## 📊 Performance

- **Upload Speed**: ~3-5 seconds per image (depends on LLaVA)
- **Search Speed**: <100ms for 1000 images (FAISS is very fast)
- **Database**: SQLite (suitable for up to ~100K images)
- **Memory**: ~1GB for 10K images in FAISS index

## 🚀 Advanced Usage

### Bulk Upload from Directory

You can manually add images to the `images/` folder and rebuild:

1. Copy images to `images/` folder
2. Run the app
3. Click "🔨 Rebuild Vector Index"

The system will process all images and generate descriptions.

### Export/Backup

Backup your data:
```bash
# Backup database
cp image_search.db image_search.backup.db

# Backup FAISS index
cp -r faiss_indexes faiss_indexes.backup
```

## 📝 License

MIT License - Feel free to use and modify!

## 🤝 Contributing

Suggestions and improvements welcome!

---

**Built with ❤️ using Streamlit, Ollama, and FAISS**
