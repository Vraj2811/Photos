#!/usr/bin/env python3
"""
Vector Inspector - View and analyze your image embeddings
"""

from pathlib import Path
import json
import numpy as np
import sqlite3

# Paths
PROJECT_ROOT = Path(__file__).parent
DATABASE_PATH = PROJECT_ROOT / "image_search.db"
FAISS_INDEX_PATH = PROJECT_ROOT / "faiss_indexes"

try:
    import faiss
except ImportError:
    print("Installing faiss for inspection...")
    import subprocess
    subprocess.run(["pip", "install", "faiss-cpu"], check=True)
    import faiss

def show_vector_details(vector, name="Vector"):
    """Show detailed statistics about a vector"""
    print(f"\n{name}:")
    print(f"  Shape: {vector.shape}")
    print(f"  Dimension: {len(vector)}")
    print(f"  Min value: {vector.min():.6f}")
    print(f"  Max value: {vector.max():.6f}")
    print(f"  Mean: {vector.mean():.6f}")
    print(f"  Std dev: {vector.std():.6f}")
    print(f"  L2 norm: {np.linalg.norm(vector):.6f}")
    print(f"  First 10 values: {vector[:10]}")

def inspect_database_vectors():
    """Inspect vectors stored in database"""
    print("=" * 70)
    print("📊 DATABASE VECTORS")
    print("=" * 70)
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM images WHERE embedding IS NOT NULL")
    total = cursor.fetchone()[0]
    print(f"\n✓ Found {total} images with embeddings\n")
    
    # Ask which image to inspect
    cursor.execute("SELECT id, filename, description FROM images WHERE embedding IS NOT NULL LIMIT 10")
    rows = cursor.fetchall()
    
    print("First 10 images:")
    for img_id, filename, desc in rows:
        print(f"  [{img_id}] {filename}")
        print(f"      {desc[:60]}...")
    
    print("\nEnter image ID to inspect (or press Enter to see first image): ", end="")
    choice = input().strip()
    
    if choice:
        img_id = int(choice)
    else:
        img_id = rows[0][0]
    
    # Get the embedding
    cursor.execute("SELECT filename, description, embedding FROM images WHERE id = ?", (img_id,))
    row = cursor.fetchone()
    
    if not row:
        print(f"❌ Image ID {img_id} not found")
        conn.close()
        return
    
    filename, description, embedding_json = row
    
    print("\n" + "=" * 70)
    print(f"IMAGE: {filename}")
    print("=" * 70)
    print(f"Description: {description}")
    
    # Parse embedding
    embedding_data = json.loads(embedding_json)
    vector = np.array(embedding_data, dtype=np.float32)
    
    show_vector_details(vector, "Embedding")
    
    conn.close()

def inspect_faiss_index():
    """Inspect FAISS index"""
    print("\n" + "=" * 70)
    print("🔢 FAISS INDEX")
    print("=" * 70)
    
    index_file = FAISS_INDEX_PATH / "vectors.index"
    mapping_file = FAISS_INDEX_PATH / "mapping.json"
    
    if not index_file.exists():
        print("❌ FAISS index not found")
        return
    
    # Load index
    index = faiss.read_index(str(index_file))
    print(f"\n✓ FAISS Index loaded")
    print(f"  Total vectors: {index.ntotal}")
    print(f"  Dimension: {index.d}")
    print(f"  Index type: {type(index).__name__}")
    
    # Load mapping
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)
    
    print(f"  ID mappings: {len(mapping)}")
    
    # Get a sample vector from index
    if index.ntotal > 0:
        # Reconstruct first vector
        vector = faiss.vector_to_array(index.reconstruct(0))
        show_vector_details(vector, "Sample vector from FAISS")

def compute_vector_similarities():
    """Compute similarities between vectors"""
    print("\n" + "=" * 70)
    print("📐 VECTOR SIMILARITIES")
    print("=" * 70)
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, filename, embedding FROM images WHERE embedding IS NOT NULL LIMIT 5")
    rows = cursor.fetchall()
    
    if len(rows) < 2:
        print("❌ Need at least 2 images to compute similarities")
        conn.close()
        return
    
    print(f"\nComputing similarities between first 5 images:\n")
    
    # Load vectors
    vectors = []
    names = []
    for img_id, filename, embedding_json in rows:
        embedding_data = json.loads(embedding_json)
        vector = np.array(embedding_data, dtype=np.float32)
        vectors.append(vector)
        names.append(f"[{img_id}] {filename[:40]}")
    
    # Compute pairwise similarities (cosine similarity)
    print("Cosine Similarity Matrix:")
    print("-" * 70)
    
    # Header
    print(f"{'':45}", end="")
    for i in range(len(names)):
        print(f"  [{i+1}]", end="")
    print()
    
    # Compute similarities
    for i, vec_i in enumerate(vectors):
        print(f"{names[i]:45}", end="")
        for j, vec_j in enumerate(vectors):
            # Cosine similarity = dot product of normalized vectors
            similarity = np.dot(vec_i, vec_j)
            if i == j:
                print(f"  1.00", end="")  # Same vector
            else:
                print(f"  {similarity:.2f}", end="")
        print()
    
    print("\nNote: Values closer to 1.0 mean more similar")
    
    conn.close()

def search_test():
    """Test search with a query"""
    print("\n" + "=" * 70)
    print("🔍 VECTOR SEARCH TEST")
    print("=" * 70)
    
    print("\nEnter a search query: ", end="")
    query = input().strip()
    
    if not query:
        print("No query entered")
        return
    
    # Generate embedding for query
    try:
        import ollama
        print(f"\n→ Generating embedding for: '{query}'")
        response = ollama.embeddings(model="nomic-embed-text", prompt=query)
        
        if 'embedding' in response:
            query_vec = np.array(response['embedding'], dtype=np.float32)
            
            # Normalize
            norm = np.linalg.norm(query_vec)
            if norm > 0:
                query_vec = query_vec / norm
            
            show_vector_details(query_vec, "Query embedding")
            
            # Load FAISS index
            index_file = FAISS_INDEX_PATH / "vectors.index"
            if not index_file.exists():
                print("❌ FAISS index not found")
                return
            
            index = faiss.read_index(str(index_file))
            
            # Search
            print(f"\n→ Searching in {index.ntotal} vectors...")
            scores, indices = index.search(query_vec.reshape(1, -1), min(5, index.ntotal))
            
            print("\nTop 5 Results:")
            print("-" * 70)
            
            # Load mapping
            mapping_file = FAISS_INDEX_PATH / "mapping.json"
            with open(mapping_file, 'r') as f:
                mapping = json.load(f)
            
            # Get image details
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            
            for i, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
                if idx >= 0:
                    image_id = mapping.get(str(int(idx)))
                    if image_id:
                        cursor.execute("SELECT filename, description FROM images WHERE id = ?", (image_id,))
                        row = cursor.fetchone()
                        if row:
                            filename, desc = row
                            print(f"\n{i}. {filename}")
                            print(f"   Confidence: {score:.4f}")
                            print(f"   Description: {desc[:100]}...")
            
            conn.close()
            
    except Exception as e:
        print(f"❌ Error: {e}")

def export_vectors_to_csv():
    """Export vectors to CSV for external analysis"""
    print("\n" + "=" * 70)
    print("💾 EXPORT VECTORS")
    print("=" * 70)
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, filename, description, embedding FROM images WHERE embedding IS NOT NULL")
    rows = cursor.fetchall()
    
    if not rows:
        print("❌ No vectors to export")
        conn.close()
        return
    
    output_file = PROJECT_ROOT / "vectors_export.csv"
    
    with open(output_file, 'w') as f:
        # Header
        f.write("id,filename,description,dimension,norm,mean,std,min,max\n")
        
        for img_id, filename, desc, embedding_json in rows:
            embedding_data = json.loads(embedding_json)
            vector = np.array(embedding_data, dtype=np.float32)
            
            # Stats
            f.write(f"{img_id},")
            f.write(f'"{filename}",')
            f.write(f'"{desc[:50]}...",')
            f.write(f"{len(vector)},")
            f.write(f"{np.linalg.norm(vector):.6f},")
            f.write(f"{vector.mean():.6f},")
            f.write(f"{vector.std():.6f},")
            f.write(f"{vector.min():.6f},")
            f.write(f"{vector.max():.6f}\n")
    
    conn.close()
    
    print(f"✅ Exported {len(rows)} vectors to: {output_file}")
    print("\nYou can open this CSV in Excel, Google Sheets, or any data analysis tool")

def main_menu():
    """Main menu"""
    while True:
        print("\n" + "=" * 70)
        print("🔍 VECTOR INSPECTOR")
        print("=" * 70)
        print("\nOptions:")
        print("  1. Inspect database vectors")
        print("  2. Inspect FAISS index")
        print("  3. Compute vector similarities")
        print("  4. Test search with query")
        print("  5. Export vectors to CSV")
        print("  6. Exit")
        print("\nChoice: ", end="")
        
        choice = input().strip()
        
        if choice == "1":
            inspect_database_vectors()
        elif choice == "2":
            inspect_faiss_index()
        elif choice == "3":
            compute_vector_similarities()
        elif choice == "4":
            search_test()
        elif choice == "5":
            export_vectors_to_csv()
        elif choice == "6":
            print("\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    print("\n🔬 Vector Inspector for AI Image Search\n")
    
    if not DATABASE_PATH.exists():
        print(f"❌ Database not found: {DATABASE_PATH}")
        exit(1)
    
    main_menu()

