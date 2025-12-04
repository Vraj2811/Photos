#!/usr/bin/env python3
"""
Rebuild FAISS index with correct embedding dimensions
Run this script to update your existing index after dimension changes.
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
import faiss
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Integer, String, Text, DateTime
import ollama

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Configuration
PROJECT_ROOT = Path(__file__).parent
DATABASE_URL = f"sqlite:///{PROJECT_ROOT}/image_search.db"
FAISS_INDEX_PATH = PROJECT_ROOT / "faiss_indexes"
NEW_EMBEDDING_DIMENSION = 768

# Database model
Base = declarative_base()

class ImageRecord(Base):
    __tablename__ = 'images'
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(Text, nullable=False)
    description = Column(Text)
    embedding = Column(Text)

def generate_embedding(text):
    """Generate embedding using Ollama"""
    try:
        response = ollama.embeddings(model='nomic-embed-text', prompt=text)
        return np.array(response['embedding'])
    except Exception as e:
        print(f"⚠️ Ollama embedding failed: {e}")
        return None

def simple_text_embedding(text, dimension=768):
    """Fallback embedding method"""
    if not text:
        text = "empty"
    
    words = text.lower().split()
    embedding = np.zeros(dimension)
    
    for i, word in enumerate(words):
        hash1 = abs(hash(word)) % dimension
        hash2 = abs(hash(word[::-1])) % dimension
        hash3 = abs(hash(word + str(len(word)))) % dimension
        
        weight = 1.0 / (i + 1)
        embedding[hash1] += weight
        embedding[hash2] += weight * 0.7
        embedding[hash3] += weight * 0.5
    
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    else:
        embedding = np.random.normal(0, 0.01, dimension)
        embedding = embedding / np.linalg.norm(embedding)
    
    return embedding

def rebuild_index():
    """Rebuild the FAISS index with correct dimensions"""
    print("🔄 Starting index rebuild...")
    
    # Connect to database
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    
    try:
        # Get all images from database
        images = session.query(ImageRecord).all()
        print(f"📊 Found {len(images)} images in database")
        
        if not images:
            print("ℹ️ No images found, nothing to rebuild")
            return
        
        # Create new FAISS index
        index = faiss.IndexFlatIP(NEW_EMBEDDING_DIMENSION)
        image_id_mapping = {}
        
        # Process each image
        successful = 0
        failed = 0
        
        for i, image in enumerate(images):
            print(f"🔄 Processing {i+1}/{len(images)}: {image.filename}")
            
            # Try to get embedding from database first
            embedding = None
            if image.embedding:
                try:
                    stored_embedding = json.loads(image.embedding)
                    if len(stored_embedding) == NEW_EMBEDDING_DIMENSION:
                        embedding = np.array(stored_embedding)
                    else:
                        print(f"  📏 Stored embedding has wrong dimension ({len(stored_embedding)}), regenerating...")
                except:
                    print("  ⚠️ Failed to parse stored embedding, regenerating...")
            
            # Generate new embedding if needed
            if embedding is None:
                if image.description:
                    # Try Ollama first
                    embedding = generate_embedding(image.description)
                    if embedding is None or len(embedding) != NEW_EMBEDDING_DIMENSION:
                        # Fallback to simple embedding
                        embedding = simple_text_embedding(image.description)
                        print(f"  📝 Used fallback embedding for {image.filename}")
                    else:
                        print(f"  ✅ Generated Ollama embedding for {image.filename}")
                    
                    # Update database with new embedding
                    try:
                        image.embedding = json.dumps(embedding.tolist())
                        session.commit()
                    except Exception as e:
                        print(f"  ⚠️ Failed to update embedding in database: {e}")
                else:
                    print(f"  ❌ No description available for {image.filename}")
                    failed += 1
                    continue
            
            # Add to FAISS index
            try:
                index.add(embedding.reshape(1, -1))
                image_id_mapping[index.ntotal - 1] = image.id
                successful += 1
            except Exception as e:
                print(f"  ❌ Failed to add to index: {e}")
                failed += 1
        
        # Save new index
        if successful > 0:
            index_file = FAISS_INDEX_PATH / "image_vectors.index"
            mapping_file = FAISS_INDEX_PATH / "id_mapping.json"
            
            # Backup old files
            if index_file.exists():
                backup_index = FAISS_INDEX_PATH / "image_vectors.index.backup"
                index_file.rename(backup_index)
                print(f"📦 Backed up old index to {backup_index}")
            
            if mapping_file.exists():
                backup_mapping = FAISS_INDEX_PATH / "id_mapping.json.backup"
                mapping_file.rename(backup_mapping)
                print(f"📦 Backed up old mapping to {backup_mapping}")
            
            # Save new files
            faiss.write_index(index, str(index_file))
            with open(mapping_file, 'w') as f:
                mapping_str = {str(k): v for k, v in image_id_mapping.items()}
                json.dump(mapping_str, f, indent=2)
            
            print(f"✅ Index rebuild complete!")
            print(f"   📊 Dimension: {NEW_EMBEDDING_DIMENSION}")
            print(f"   ✅ Successful: {successful}")
            print(f"   ❌ Failed: {failed}")
            print(f"   💾 Index saved to: {index_file}")
            print(f"   🗂️ Mapping saved to: {mapping_file}")
        else:
            print("❌ No embeddings were successfully generated")
    
    except Exception as e:
        print(f"❌ Error during rebuild: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    print("🚀 FAISS Index Rebuild Tool")
    print(f"📏 Target dimension: {NEW_EMBEDDING_DIMENSION}")
    print()
    
    # Check if Ollama is available
    try:
        models = ollama.list()
        print("✅ Ollama connection successful")
        if any('nomic-embed-text' in str(model) for model in models.models):
            print("✅ nomic-embed-text model available")
        else:
            print("⚠️ nomic-embed-text model not found, will use fallback embeddings")
    except Exception as e:
        print(f"⚠️ Ollama not available: {e}")
        print("📝 Will use fallback text embeddings")
    
    print()
    confirm = input("🤔 Do you want to rebuild the index? This will update all embeddings. (y/N): ")
    
    if confirm.lower().startswith('y'):
        rebuild_index()
    else:
        print("ℹ️ Index rebuild cancelled")