# 🔬 Vector Inspector Guide

The **Vector Inspector** is now built into your Gradio app! Access it from the 4th tab.

## 🚀 Quick Start

```bash
python3 gradio_app.py
```

Then click on the **"🔬 Vector Inspector"** tab.

---

## 🎯 Features

### 1. 🔍 **Inspect Vector**

View detailed information about any image's embedding vector.

**What you see:**
- Vector dimension (should be 768)
- L2 Norm (should be ~1.0 for normalized vectors)
- Statistical measures (mean, std dev, min, max)
- First 20 vector values
- Health checks (dimension, normalization, NaN/Inf checks)

**How to use:**
1. Select an image from dropdown
2. Click "🔍 Inspect"
3. View detailed vector analysis

**Use case:** Verify that embeddings are correctly generated and normalized.

---

### 2. 📊 **Compare Vectors**

Compare similarity between two image vectors.

**What you see:**
- **Cosine Similarity** (1.0 = identical, -1.0 = opposite)
  - 🟢 > 0.8 = Very Similar
  - 🟡 0.5-0.8 = Somewhat Similar
  - 🔴 < 0.5 = Not Similar
- **Euclidean Distance** (0.0 = identical)
- Difference statistics
- Top 5 most different dimensions

**How to use:**
1. Select Image 1
2. Select Image 2
3. Click "📊 Compare"
4. See how similar they are

**Use cases:**
- Find duplicate or very similar images
- Understand why search returns certain results
- Validate that similar images have similar vectors

---

### 3. 🧪 **Test Query**

Generate and analyze embedding for any text query.

**What you see:**
- Query vector statistics
- First 20 vector values
- **Search Preview** - Top 5 matching images automatically

**How to use:**
1. Enter a query (e.g., "Nike shoes")
2. Click "🧪 Test Query"
3. See the query's embedding details
4. Preview which images match

**Use cases:**
- Test different query phrasings
- Understand query embeddings
- Debug why searches return unexpected results
- Find optimal query wording

---

### 4. 💾 **Export**

Export all vectors to CSV for external analysis.

**What you get:**
A `vectors_export.csv` file with:
- Image ID
- Filename
- Description
- Dimension
- Norm
- Mean, Std, Min, Max

**How to use:**
1. Click "💾 Export to CSV"
2. File saved to project root
3. Open in Excel, Google Sheets, Python, R, etc.

**Use cases:**
- Analyze vectors in external tools
- Create visualizations (t-SNE, PCA)
- Statistical analysis
- Machine learning experiments

---

## 🎯 Common Use Cases

### Debugging Search Results

**Scenario:** Search returns unexpected results

**Steps:**
1. Go to **🧪 Test Query** tab
2. Enter your search query
3. View the query vector and preview results
4. If results are wrong, try different query phrasing
5. Use **📊 Compare Vectors** to see why certain images matched

---

### Finding Similar Images

**Scenario:** Want to find images similar to a specific one

**Steps:**
1. Note the filename of your target image
2. Go to **📊 Compare Vectors** tab
3. Select your target image as Image 1
4. Try different images as Image 2
5. Look for cosine similarity > 0.7

---

### Verifying Data Quality

**Scenario:** Want to ensure all embeddings are correct

**Steps:**
1. Go to **🔍 Inspect Vector** tab
2. Check a few random images
3. Verify:
   - ✅ Dimension = 768
   - ✅ Norm ≈ 1.0
   - ✅ No NaN/Inf values
   - ✅ Reasonable mean/std values

---

### Analyzing Vector Distribution

**Scenario:** Want to understand your vector space

**Steps:**
1. Go to **💾 Export** tab
2. Click "Export to CSV"
3. Open `vectors_export.csv`
4. Analyze statistics:
   - All norms should be ~1.0
   - Mean should be close to 0
   - Std should be reasonable (0.1-0.3)

---

## 📊 Understanding the Numbers

### Cosine Similarity
```
1.0  = Identical vectors
0.8  = Very similar (same topic)
0.6  = Somewhat similar
0.4  = Different but related
0.2  = Very different
-1.0 = Complete opposite
```

### L2 Norm (for normalized vectors)
```
~1.0 = Properly normalized ✅
>1.1 = Not normalized ⚠️
<0.9 = Not normalized ⚠️
```

### Euclidean Distance
```
0.0  = Identical
<1.0 = Very similar
1.0-2.0 = Somewhat similar
>2.0 = Different
```

---

## 💡 Pro Tips

### 1. **Test Queries Before Searching**
Use the Test Query tab to see if your query will work before running a full search.

### 2. **Compare Top Results**
After a search, compare the top results to understand why they ranked high.

### 3. **Find Threshold**
Use Compare Vectors to find the similarity threshold that works for your use case.

### 4. **Export for Visualization**
Export vectors and use tools like Python's matplotlib or seaborn to visualize the vector space with t-SNE or PCA.

### 5. **Spot Check Quality**
Regularly inspect random vectors to ensure embedding quality stays consistent.

---

## 🔧 Advanced Usage

### Python Analysis

After exporting, analyze with Python:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load exported vectors
df = pd.read_csv('vectors_export.csv')

# Check norms (should all be ~1.0)
plt.hist(df['norm'], bins=50)
plt.title('Vector Norms Distribution')
plt.show()

# Check means (should center around 0)
plt.hist(df['mean'], bins=50)
plt.title('Vector Means Distribution')
plt.show()
```

### Finding Outliers

Images with unusual vectors might have issues:

```python
# Find images with non-normalized vectors
outliers = df[abs(df['norm'] - 1.0) > 0.1]
print("Potential issues:")
print(outliers[['id', 'filename', 'norm']])
```

---

## ❓ FAQ

**Q: What dimension should vectors be?**  
A: All should be 768 (EMBEDDING_DIMENSION)

**Q: What's a good cosine similarity for "similar" images?**  
A: > 0.7 typically indicates strong similarity

**Q: Why is my vector not normalized?**  
A: Should be ~1.0. If not, run `python3 fix_embeddings.py`

**Q: How do I visualize vectors?**  
A: Export to CSV, then use Python with scikit-learn's t-SNE or PCA

**Q: Can I modify vectors?**  
A: Not directly in UI. Export, modify, then re-import to database

---

## 🎨 Example Workflow

### Complete Image Search Analysis

1. **Upload Images** (Upload tab)
   - Add 10-20 images
   
2. **Verify Quality** (Vector Inspector → Inspect)
   - Check 3-4 random images
   - Ensure all pass health checks
   
3. **Test Query** (Vector Inspector → Test Query)
   - Try: "person wearing Nike"
   - Note which images appear
   
4. **Compare Results** (Vector Inspector → Compare)
   - Compare top 2 results
   - Check similarity score
   
5. **Search** (Search tab)
   - Run actual search
   - Results should match test
   
6. **Export** (Vector Inspector → Export)
   - Export for documentation
   - Analyze in external tools

---

## 🚀 Next Steps

Now that you have vector inspection built in:

1. **Test your search** - Use Test Query before searching
2. **Validate results** - Compare vectors to understand rankings
3. **Monitor quality** - Regularly inspect random vectors
4. **Optimize queries** - Test different phrasings
5. **Analyze patterns** - Export and visualize vector space

---

**Pro Tip:** The Vector Inspector is also available as a standalone CLI tool (`python3 inspect_vectors.py`) if you prefer terminal-based analysis!

---

## 📞 Quick Reference

| Task | Tab | Action |
|------|-----|--------|
| View vector details | 🔍 Inspect | Select image → Inspect |
| Compare 2 images | 📊 Compare | Select 2 images → Compare |
| Test search query | 🧪 Test Query | Enter text → Test |
| Export all vectors | 💾 Export | Click Export |

---

**Enjoy exploring your vector space!** 🔬🎉

