# 📦 Bulk Processing Guide

This guide shows you how to process many images at once and build their vectors.

## 🚀 Quick Method: Using the Script

### Step 1: Add Images to Folder

Copy all your images to the `images` folder:

```bash
# From your terminal
cp /path/to/your/images/*.jpg images/
cp /path/to/your/images/*.png images/
```

Or manually drag and drop images into the `images` folder.

### Step 2: Make Sure Ollama is Running

```bash
# Terminal 1 - Start Ollama (keep this running)
ollama serve
```

### Step 3: Run the Vector Builder Script

```bash
# In a new terminal, go to project folder
cd /Users/salam9/Desktop/Image_Search

# Run the script
python build_vectors.py
```

The script will:
1. ✅ Check Ollama connection
2. ✅ Scan the `images` folder
3. ✅ Process each new image (skip already processed ones)
4. ✅ Generate descriptions with LLaVA
5. ✅ Create embeddings
6. ✅ Store in database
7. ✅ Build FAISS index
8. ✅ Show progress for each image

### Step 4: Start Searching!

```bash
streamlit run streamlit_app.py
```

Go to Search tab and try searching!

## 📊 What the Script Does

```
For each image in images/:
  1. Check if already in database (skip if yes)
  2. Generate 2-3 line description with LLaVA
  3. Convert description to 768-dim embedding
  4. Save to database with embedding
  5. Add to FAISS vector index

Final: Build complete FAISS index for fast search
```

## 💡 Example Output

```
==================================================================
🔨 Image Vector Builder
==================================================================
✓ Ollama connected. Available models: llava, nomic-embed-text
✓ Both required models available

→ Initializing database...
✓ Database has 0 existing images
→ Scanning images...
✓ Found 10 image files
→ Processing 10 new images...

[1/10] Processing: sunset.jpg
  → Generating description...
  ✓ Description: A beautiful sunset over the ocean with orange and pink...
  → Generating embedding...
  ✓ Embedding generated (768 dimensions)
  ✓ Saved to database (ID: 1)

[2/10] Processing: mountain.jpg
  ...

==================================================================
✅ Processing complete!
   Successfully processed: 10/10 images
==================================================================

→ Building FAISS index...
  → Indexing 10 images...
  ✓ FAISS index built with 10 vectors
  ✓ Saved to faiss_indexes/vectors.index

==================================================================
🎉 All done! Your images are now searchable.
==================================================================
```

## 🔧 Troubleshooting

### "Ollama connection failed"
```bash
# Make sure Ollama is running
ollama serve
```

### "Model not found"
```bash
ollama pull llava
ollama pull nomic-embed-text
```

### "No images found"
- Check images are in the `images/` folder
- Supported formats: JPG, JPEG, PNG, WebP, BMP

### Script fails midway
- Already processed images won't be reprocessed
- Just run the script again, it will continue from where it stopped

### Want to reprocess all images
```bash
# Delete database and index
rm image_search.db
rm -rf faiss_indexes/*

# Run script again
python build_vectors.py
```

## 📁 File Structure

```
Image_Search/
├── images/              # Put your images here
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── build_vectors.py     # Run this script
├── image_search.db      # Database (auto-created)
└── faiss_indexes/       # Vector index (auto-created)
    ├── vectors.index
    └── mapping.json
```

## ⚡ Performance

- **Speed**: ~3-5 seconds per image
  - Description: ~2-3 seconds (LLaVA)
  - Embedding: ~1 second
  - Database: instant

- **Batch Size**: Can process hundreds of images
  - 10 images: ~30-50 seconds
  - 100 images: ~5-8 minutes
  - 1000 images: ~50-80 minutes

## 🎯 Best Practices

### For Large Batches:
1. Start with a small test (5-10 images)
2. Verify they show up in search
3. Then process the full batch

### Image Naming:
- Use descriptive filenames
- No special characters that might cause issues
- Keep filenames reasonable length

### Image Quality:
- Any size works (will be processed as-is)
- Clearer images = better descriptions
- Common formats work best (JPG, PNG)

## 🔄 Alternative: Use the UI

If you prefer, you can also upload through the Streamlit UI:

1. Run `streamlit run streamlit_app.py`
2. Go to Upload tab
3. Select images (can select multiple)
4. Click "Process All Images"

**Script is faster for bulk processing (100+ images)**  
**UI is better for occasional uploads**

## 📝 Notes

- Script automatically skips already-processed images
- Safe to run multiple times
- Creates database and index if they don't exist
- Normalizes all embeddings for accurate search
- Builds optimized FAISS index at the end

---

**Need help?** Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

