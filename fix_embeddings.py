#!/usr/bin/env python3
"""
Fix embeddings for existing images in database
Regenerates embeddings from descriptions without re-processing images
"""

from pathlib import Path
import json
import numpy as np
from datetime import datetime
import time

try:
    import ollama
    import faiss
    from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
    from sqlalchemy.orm import declarative_base, sessionmaker
except ImportError as e:
    print(f"❌ Missing packages: {e}")
    exit(1)

# Configuration
PROJECT_ROOT = Path(__file__).parent
DATABASE_PATH = PROJECT_ROOT / "image_search.db"
FAISS_INDEX_PATH = PROJECT_ROOT / "faiss_indexes"
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIMENSION = 768
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

FAISS_INDEX_PATH.mkdir(exist_ok=True)

# Database setup
Base = declarative_base()

class ImageRecord(Base):
    __tablename__ = 'images'
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(Text, nullable=False)
    description = Column(Text)
    embedding = Column(Text)
    created_at = Column(DateTime, default=datetime.now)

def init_database():
    engine = create_engine(DATABASE_URL, echo=False)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal

def generate_embedding(text):
    """Generate embedding from text"""
    try:
        response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
        
        if 'embedding' in response and response['embedding']:
            embedding = np.array(response['embedding'], dtype=np.float32)
            
            # Ensure correct dimension
            if len(embedding) > EMBEDDING_DIMENSION:
                embedding = embedding[:EMBEDDING_DIMENSION]
            elif len(embedding) < EMBEDDING_DIMENSION:
                padded = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
                padded[:len(embedding)] = embedding
                embedding = padded
            
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
        return None
    except Exception as e:
        print(f"  ❌ Embedding failed: {e}")
        return None

def fix_embeddings():
    """Fix all embeddings in database"""
    print("=" * 70)
    print("🔧 Embedding Dimension Fixer")
    print("=" * 70)
    print("\nThis will regenerate embeddings for images with wrong dimensions")
    print("(from their existing descriptions, without re-processing images)\n")
    
    # Initialize database
    SessionLocal = init_database()
    session = SessionLocal()
    
    # Get all images
    all_images = session.query(ImageRecord).all()
    print(f"✓ Found {len(all_images)} images in database")
    
    # Check which need fixing
    needs_fixing = []
    correct = []
    missing = []
    
    for img in all_images:
        if not img.embedding:
            missing.append(img)
        else:
            try:
                embedding_data = json.loads(img.embedding)
                dim = len(embedding_data)
                if dim != EMBEDDING_DIMENSION:
                    needs_fixing.append(img)
                else:
                    correct.append(img)
            except:
                needs_fixing.append(img)
    
    print(f"  ✅ Correct embeddings: {len(correct)}")
    print(f"  ⚠️  Need fixing: {len(needs_fixing)}")
    print(f"  ❌ Missing embeddings: {len(missing)}")
    
    if not needs_fixing and not missing:
        print("\n✅ All embeddings are already correct!")
        session.close()
        return
    
    # Fix embeddings
    to_fix = needs_fixing + missing
    print(f"\n→ Fixing {len(to_fix)} embeddings...\n")
    
    success_count = 0
    error_count = 0
    
    for i, img in enumerate(to_fix, 1):
        print(f"[{i}/{len(to_fix)}] {img.filename}")
        
        if not img.description:
            print(f"  ❌ No description available - skipping")
            error_count += 1
            continue
        
        # Generate new embedding from description
        print(f"  → Generating embedding from description...")
        embedding = generate_embedding(img.description)
        
        if embedding is None:
            print(f"  ❌ Failed to generate embedding")
            error_count += 1
            continue
        
        # Verify dimension
        if len(embedding) != EMBEDDING_DIMENSION:
            print(f"  ❌ Wrong dimension: {len(embedding)} != {EMBEDDING_DIMENSION}")
            error_count += 1
            continue
        
        # Update database
        try:
            embedding_json = json.dumps(embedding.tolist())
            img.embedding = embedding_json
            session.commit()
            print(f"  ✅ Updated! ({len(embedding)} dimensions)")
            success_count += 1
        except Exception as e:
            session.rollback()
            print(f"  ❌ Database error: {e}")
            error_count += 1
        
        # Small delay to avoid overwhelming Ollama
        if i % 10 == 0:
            time.sleep(1)
    
    session.close()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print(f"  ✅ Successfully fixed: {success_count}")
    print(f"  ❌ Errors: {error_count}")
    print("=" * 70)
    
    if success_count > 0:
        print("\n→ Now rebuilding FAISS index...")
        rebuild_faiss_index()

def rebuild_faiss_index():
    """Rebuild FAISS index from database"""
    try:
        SessionLocal = init_database()
        session = SessionLocal()
        
        all_images = session.query(ImageRecord).all()
        
        if not all_images:
            print("  ⚠️  No images to index")
            session.close()
            return
        
        print(f"  → Building index for {len(all_images)} images...")
        
        # Create new index
        index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        id_mapping = {}
        
        vectors_to_add = []
        image_ids = []
        skipped = 0
        
        for img in all_images:
            if img.embedding:
                try:
                    embedding_data = json.loads(img.embedding)
                    embedding = np.array(embedding_data, dtype=np.float32)
                    
                    # Verify dimension
                    if len(embedding) != EMBEDDING_DIMENSION:
                        print(f"    ⚠️  Skipping image {img.id}: wrong dimension {len(embedding)}")
                        skipped += 1
                        continue
                    
                    # Normalize
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                    
                    vectors_to_add.append(embedding)
                    image_ids.append(img.id)
                    
                except Exception as e:
                    print(f"    ⚠️  Error loading embedding for image {img.id}: {e}")
                    skipped += 1
        
        if vectors_to_add:
            vectors_array = np.vstack(vectors_to_add).astype(np.float32)
            index.add(vectors_array)
            
            # Create mapping
            for idx, image_id in enumerate(image_ids):
                id_mapping[idx] = image_id
            
            # Save to disk
            index_file = FAISS_INDEX_PATH / "vectors.index"
            mapping_file = FAISS_INDEX_PATH / "mapping.json"
            
            faiss.write_index(index, str(index_file))
            
            with open(mapping_file, 'w') as f:
                mapping_str = {str(k): v for k, v in id_mapping.items()}
                json.dump(mapping_str, f, indent=2)
            
            print(f"  ✅ Index built with {len(vectors_to_add)} vectors")
            if skipped > 0:
                print(f"  ⚠️  Skipped {skipped} images with issues")
        else:
            print("  ⚠️  No valid embeddings to index")
        
        session.close()
        
    except Exception as e:
        print(f"  ❌ FAISS rebuild failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("\n🔧 Embedding Dimension Fixer\n")
    
    try:
        # Check Ollama
        print("Checking Ollama connection...")
        try:
            models_response = ollama.list()
            print("✓ Ollama connected")
        except Exception as e:
            print(f"❌ Ollama not available: {e}")
            print("\nMake sure Ollama is running: ollama serve")
            exit(1)
        
        # Run fix
        fix_embeddings()
        
        print("\n" + "=" * 70)
        print("🎉 All done! Your embeddings are now fixed.")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Run: streamlit run streamlit_app.py")
        print("   or: python3 gradio_app.py")
        print("2. Start searching!\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

