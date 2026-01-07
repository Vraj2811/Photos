import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import ollama
import numpy as np
import base64
import json
import cv2
import io
from PIL import Image
from insightface.app import FaceAnalysis
from config import VISION_MODEL, EMBEDDING_MODEL, EMBEDDING_DIMENSION

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

class OllamaProcessor:
    def __init__(self):
        self.vision_model = VISION_MODEL
        self.embedding_model = EMBEDDING_MODEL
        self.models_available = []
        self.check_models()
    
    def check_models(self):
        try:
            models_response = ollama.list()
            if hasattr(models_response, 'models'):
                for model in models_response.models:
                    if hasattr(model, 'model'):
                        self.models_available.append(model.model.split(':')[0])
            return True
        except:
            return False
    
    def generate_description(self, image_input):
        try:
            images = []
            if isinstance(image_input, bytes):
                # Convert bytes to base64 string
                base64_image = base64.b64encode(image_input).decode('utf-8')
                images.append(base64_image)
            else:
                # Path string
                images.append(image_input)

            response = ollama.chat(
                model=self.vision_model,
                messages=[{
                    'role': 'user',
                    'content': 'Describe this image in 2-3 clear, concise sentences.',
                    'images': images
                }]
            )
            return response['message']['content'].strip()
        except Exception as e:
            print(f"Description failed: {e}")
            return None
    
    def generate_embedding(self, text):
        try:
            response = ollama.embeddings(model=self.embedding_model, prompt=text)
            if 'embedding' in response and response['embedding']:
                embedding = np.array(response['embedding'], dtype=np.float32)
                
                if len(embedding) > EMBEDDING_DIMENSION:
                    embedding = embedding[:EMBEDDING_DIMENSION]
                elif len(embedding) < EMBEDDING_DIMENSION:
                    padded = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
                    padded[:len(embedding)] = embedding
                    embedding = padded
                
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                
                return embedding
            return None
        except Exception as e:
            print(f"Embedding failed: {e}")
            return None

class FaceProcessor:
    def __init__(self, db):
        print("Initializing FaceAnalysis(name='antelopev2')...")
        self.app = FaceAnalysis(name="antelopev2", providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.similarity_threshold = 0.6
        self.db = db

    def detect_faces(self, image_bytes):
        """Detect faces and return face data without saving to DB."""
        try:
            # Convert bytes to cv2 image
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return []
            
            faces = self.app.get(img)
            return faces
        except Exception as e:
            print(f"Face detection failed: {e}")
            return []

    def save_faces(self, faces, image_id):
        """Save detected faces to DB and link to image_id."""
        try:
            print(f"Saving {len(faces)} faces for image {image_id}")
            for face in faces:
                embedding = face.normed_embedding
                bbox = face.bbox
                confidence = face.det_score
                
                # Find matching group
                group_id = self.find_matching_group(embedding)
                
                if group_id is None:
                    # Create new group
                    group_id = self.db.add_face_group(embedding)
                    print(f"Created new face group {group_id}")
                
                # Save detected face
                self.db.add_detected_face(image_id, group_id, embedding, bbox, confidence)
        except Exception as e:
            print(f"Failed to save faces for image {image_id}: {e}")

    def process_image(self, image_bytes, image_id):
        """Legacy method for background tasks."""
        faces = self.detect_faces(image_bytes)
        if faces:
            self.save_faces(faces, image_id)

    def find_matching_group(self, face_embedding):
        face_embedding = normalize(face_embedding)

        groups = self.db.get_all_face_groups()
        best_match_id = None
        best_similarity = -1

        for group in groups:
            group_embedding = normalize(
                json.loads(group.representative_embedding)
            )

            similarity = np.dot(face_embedding, group_embedding)

            if similarity > self.similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match_id = group.id

        return best_match_id
