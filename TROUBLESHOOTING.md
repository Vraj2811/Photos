# 🔧 Troubleshooting Guide

## Search Not Working? Follow These Steps

### Step 1: Check System Status (Sidebar)

Look at the sidebar stats:
- **Images**: Should show count > 0
- **Vectors**: Should match Images count

If **Vectors = 0** but **Images > 0**:
1. Click "🔨 Rebuild Vector Index" button
2. Wait for completion
3. Try searching again

### Step 2: Verify Ollama is Running

In sidebar, check "🤖 AI Models" section:
- Should see ✅ next to both models
- If you see ❌, follow the commands shown

**Fix:**
```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Pull models if missing
ollama pull llava
ollama pull nomic-embed-text
```

### Step 3: Check Debug Info

In sidebar, expand "🔍 Debug Info":
- **Index File Exists**: Should be `True`
- **FAISS Index Total**: Should match vector count

If files don't exist or count is 0:
1. Upload at least one image first
2. Or rebuild index if images exist

### Step 4: Test with Simple Query

Try a very simple search:
- Type: "image" or "photo"
- Should return results if database has any images

If no results:
- Database might be empty
- Embeddings might not be generated
- Rebuild index and try again

## Common Issues & Solutions

### Issue 1: "No images in the database yet"
**Cause:** Database is empty  
**Solution:** Go to Upload tab and upload images

### Issue 2: "Failed to generate embedding for query"
**Cause:** Ollama not running or embedding model missing  
**Solution:**
```bash
# Start Ollama
ollama serve

# In new terminal, pull embedding model
ollama pull nomic-embed-text
```

### Issue 3: Search returns no results but images exist
**Cause:** Vector index not built or out of sync  
**Solution:**
1. Go to sidebar
2. Click "🔨 Rebuild Vector Index"
3. Wait for "✅ Rebuilt X vectors!"
4. Try search again

### Issue 4: "Vector search error" message
**Cause:** FAISS index corruption or dimension mismatch  
**Solution:**
```bash
# Delete old indexes
rm -rf faiss_indexes/*

# Restart app (it will auto-rebuild)
streamlit run streamlit_app.py
```

### Issue 5: Upload works but search doesn't find images
**Cause:** Embeddings weren't generated during upload  
**Solution:**
1. Check if Ollama was running during upload
2. Rebuild index to regenerate embeddings
3. Re-upload images if needed

### Issue 6: Confidence scores are all very low (<0.3)
**Cause:** Query doesn't match image descriptions well  
**Solution:**
- Try more specific queries
- Check image descriptions in Gallery tab
- Use terms that appear in descriptions

## Testing the System

### Test 1: Verify Upload
1. Upload one test image
2. Check sidebar: Images should increase by 1
3. Check sidebar: Vectors should increase by 1
4. If they match, upload is working ✅

### Test 2: Verify Index
1. Go to Gallery tab
2. Find an image and read its description
3. Go to Search tab
4. Type a word from that description
5. Image should appear in results ✅

### Test 3: Verify Ollama
```bash
# Test Ollama is responding
ollama list

# Should show both models:
# - llava
# - nomic-embed-text
```

### Test 4: Check Files
```bash
# Check images are saved
ls images/
# Should show image files

# Check database exists
ls -lh image_search.db
# Should show database file

# Check FAISS index exists
ls -lh faiss_indexes/
# Should show vectors.index and mapping.json
```

## Manual Index Rebuild

If automatic rebuild doesn't work:

1. **Delete old index:**
   ```bash
   rm -rf faiss_indexes/*
   ```

2. **Restart Streamlit:**
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Rebuild in app:**
   - Go to sidebar
   - Click "🔨 Rebuild Vector Index"
   - Wait for completion

## Check Logs

Run Streamlit with verbose output:
```bash
streamlit run streamlit_app.py --logger.level=debug
```

Look for error messages related to:
- Ollama connection
- Embedding generation
- FAISS operations

## Still Not Working?

### Quick Reset (Nuclear Option)
```bash
# Backup your images first!
cp -r images images_backup

# Clean everything
rm -rf faiss_indexes/*
rm image_search.db

# Restart
streamlit run streamlit_app.py

# Re-upload images
```

### Verify Installation
```bash
# Check Python packages
pip list | grep -E "streamlit|ollama|faiss|pillow|numpy|pandas|sqlalchemy"

# Reinstall if needed
pip install -r requirements.txt --upgrade
```

### Check Ollama Installation
```bash
# Verify Ollama is installed
ollama --version

# Check if Ollama is running
curl http://localhost:11434/api/tags

# Should return JSON with model list
```

## Getting Help

If you're still stuck, check these details:

1. **System Status** (from sidebar):
   - Images count
   - Vectors count
   - Model status

2. **Debug Info** (from sidebar expander):
   - All file paths
   - File existence status
   - Index totals

3. **Terminal Output**:
   - Any error messages
   - Ollama connection status
   - Embedding generation logs

4. **Test Query**:
   - What query did you try?
   - What error message appeared?
   - Do images exist in database?

## Prevention Tips

✅ **Always start Ollama before using the app**
```bash
ollama serve
```

✅ **Verify models are installed**
```bash
ollama list
```

✅ **Check sidebar status before searching**
- Green checkmarks = good to go
- Red X marks = needs attention

✅ **Rebuild index after:**
- Manually adding images to folder
- Ollama was offline during upload
- System errors or crashes
- Database operations

✅ **Keep uploads and searches separate**
- Upload all images first
- Then search after uploads complete
- Don't mix operations

## Performance Notes

**Normal behavior:**
- First upload: 3-5 seconds per image
- Search: <1 second for most queries
- Index rebuild: ~1 second per 100 images

**If slower:**
- Check Ollama CPU/GPU usage
- Verify models are loaded in Ollama
- Large images (>5MB) take longer

---

**Most search issues are solved by:**
1. Starting Ollama (`ollama serve`)
2. Rebuilding the index (sidebar button)
3. Making sure images were uploaded successfully

