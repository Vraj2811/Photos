#!/usr/bin/env python3
"""
Diagnostic script to check database and FAISS index status
"""

from pathlib import Path
import json
import sqlite3
import numpy as np

# Paths
PROJECT_ROOT = Path(__file__).parent
DATABASE_PATH = PROJECT_ROOT / "image_search.db"
FAISS_INDEX_PATH = PROJECT_ROOT / "faiss_indexes"
EXPECTED_DIM = 768

print("=" * 70)
print("🔍 Database & Index Diagnostic Tool")
print("=" * 70)

# Check database
print("\n📊 DATABASE STATUS:")
print("-" * 70)

if not DATABASE_PATH.exists():
    print("❌ Database file not found!")
else:
    print(f"✅ Database found: {DATABASE_PATH}")
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Count images
    cursor.execute("SELECT COUNT(*) FROM images")
    total_images = cursor.fetchone()[0]
    print(f"   Total images: {total_images}")
    
    # Check embeddings
    cursor.execute("SELECT id, filename, description, embedding FROM images")
    rows = cursor.fetchall()
    
    valid_embeddings = 0
    invalid_embeddings = 0
    missing_embeddings = 0
    
    print("\n📋 IMAGE DETAILS:")
    print("-" * 70)
    
    for row in rows:
        img_id, filename, description, embedding_json = row
        print(f"\n🖼️  ID: {img_id} - {filename}")
        print(f"   Description: {description[:80] if description else 'None'}...")
        
        if not embedding_json:
            print(f"   ❌ No embedding stored")
            missing_embeddings += 1
        else:
            try:
                embedding_data = json.loads(embedding_json)
                embedding = np.array(embedding_data)
                dim = len(embedding)
                
                if dim == EXPECTED_DIM:
                    print(f"   ✅ Embedding: {dim} dimensions (correct)")
                    valid_embeddings += 1
                else:
                    print(f"   ⚠️  Embedding: {dim} dimensions (expected {EXPECTED_DIM})")
                    invalid_embeddings += 1
                    
            except Exception as e:
                print(f"   ❌ Embedding error: {e}")
                invalid_embeddings += 1
    
    conn.close()
    
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print(f"  ✅ Valid embeddings:   {valid_embeddings}")
    print(f"  ⚠️  Invalid embeddings: {invalid_embeddings}")
    print(f"  ❌ Missing embeddings: {missing_embeddings}")
    print("=" * 70)

# Check FAISS index
print("\n🔢 FAISS INDEX STATUS:")
print("-" * 70)

index_file = FAISS_INDEX_PATH / "vectors.index"
mapping_file = FAISS_INDEX_PATH / "mapping.json"

if not index_file.exists():
    print("❌ FAISS index file not found")
else:
    print(f"✅ Index file found: {index_file}")
    
    try:
        import faiss
        index = faiss.read_index(str(index_file))
        print(f"   Vectors in index: {index.ntotal}")
        print(f"   Index dimension: {index.d}")
        
        if index.d != EXPECTED_DIM:
            print(f"   ⚠️  WARNING: Index dimension {index.d} != expected {EXPECTED_DIM}")
        
    except Exception as e:
        print(f"   ❌ Failed to read index: {e}")

if not mapping_file.exists():
    print("❌ Mapping file not found")
else:
    print(f"✅ Mapping file found: {mapping_file}")
    
    try:
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
            print(f"   Mappings: {len(mapping)}")
    except Exception as e:
        print(f"   ❌ Failed to read mapping: {e}")

# Recommendations
print("\n💡 RECOMMENDATIONS:")
print("-" * 70)

if total_images == 0:
    print("➡️  Upload some images to get started")
elif missing_embeddings > 0 or invalid_embeddings > 0:
    print("➡️  Run: python build_vectors.py")
    print("   This will regenerate embeddings for all images")
elif valid_embeddings > 0:
    if not index_file.exists():
        print("➡️  FAISS index missing. Click 'Rebuild Index' in the app")
    elif index.ntotal != valid_embeddings:
        print(f"➡️  Index out of sync ({index.ntotal} vectors vs {valid_embeddings} images)")
        print("   Click 'Rebuild Index' in the app")
    else:
        print("✅ Everything looks good! You're ready to search.")

print("\n" + "=" * 70)

