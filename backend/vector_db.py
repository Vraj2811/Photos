import faiss
import json
import numpy as np
from config import FAISS_INDEX_PATH, EMBEDDING_DIMENSION

class VectorDB:
    def __init__(self):
        self.index_file = FAISS_INDEX_PATH / "vectors.index"
        self.mapping_file = FAISS_INDEX_PATH / "mapping.json"
        
        # Ensure directory exists
        FAISS_INDEX_PATH.mkdir(parents=True, exist_ok=True)
        
        self.index = None
        self.id_mapping = {}
        self.load_or_create_index()

    
    def load_or_create_index(self):
        if self.index_file.exists() and self.mapping_file.exists():
            try:
                self.index = faiss.read_index(str(self.index_file))
                with open(self.mapping_file, 'r') as f:
                    mapping_data = json.load(f)
                    self.id_mapping = {int(k): v for k, v in mapping_data.items()}
                print(f"Loaded index with {self.index.ntotal} vectors")
            except:
                self.create_new_index()
        else:
            self.create_new_index()
    
    def create_new_index(self):
        self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        self.id_mapping = {}
    
    def add_vector(self, embedding, image_id):
        try:
            embedding = embedding.astype(np.float32).reshape(1, -1)
            self.index.add(embedding)
            faiss_idx = self.index.ntotal - 1
            self.id_mapping[faiss_idx] = image_id
            self.save_index()
            return True
        except Exception as e:
            print(f"Failed to add vector: {e}")
            return False
    
    def search(self, query_embedding, top_k=5):
        try:
            if self.index.ntotal == 0:
                return []
            
            query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
            k = min(top_k, self.index.ntotal)
            scores, indices = self.index.search(query_embedding, k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and int(idx) in self.id_mapping:
                    results.append({
                        'image_id': self.id_mapping[int(idx)],
                        'confidence': float(score)
                    })
            return results
        except:
            return []
    
    def delete_vector(self, image_id):
        try:
            # Find faiss index for image_id
            faiss_idx = -1
            for k, v in self.id_mapping.items():
                if v == image_id:
                    faiss_idx = k
                    break
            
            if faiss_idx != -1:
                # Remove from FAISS
                self.index.remove_ids(np.array([faiss_idx], dtype=np.int64))
                
                # Update mapping (shift indices)
                new_mapping = {}
                for k, v in self.id_mapping.items():
                    if k < faiss_idx:
                        new_mapping[k] = v
                    elif k > faiss_idx:
                        new_mapping[k - 1] = v
                self.id_mapping = new_mapping
                
                self.save_index()
                print(f"Deleted vector for image {image_id} (index {faiss_idx})")
                return True
            return False
        except Exception as e:
            print(f"Failed to delete vector: {e}")
            return False

    def save_index(self):
        try:
            faiss.write_index(self.index, str(self.index_file))
            with open(self.mapping_file, 'w') as f:
                mapping_str = {str(k): v for k, v in self.id_mapping.items()}
                json.dump(mapping_str, f)
        except Exception as e:
            print(f"Save failed: {e}")
