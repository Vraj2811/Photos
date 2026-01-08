from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import json
from datetime import datetime
from config import DATABASE_URL
from models import Base, ImageRecord, FaceGroup, DetectedFace

# Initialize database
engine = create_engine(DATABASE_URL, echo=False)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

class ImageDB:
    def __init__(self):
        self.SessionLocal = SessionLocal
    
    def add_image(self, filename, file_path, description, embedding, **kwargs):
        session = self.SessionLocal()
        try:
            if embedding is not None:
                embedding_json = json.dumps(embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding))
            else:
                embedding_json = None
                
            image_record = ImageRecord(
                filename=filename,
                file_path=file_path,
                drive_file_id=kwargs.get('drive_file_id'),
                description=description,
                embedding=embedding_json,
                created_at=datetime.now()
            )
            session.add(image_record)
            session.commit()
            return image_record.id
        except Exception as e:
            session.rollback()
            print(f"Database error: {e}")
            return None
        finally:
            session.close()
    
    def get_all_images(self, limit=None, offset=None, exclude_videos=True):
        session = self.SessionLocal()
        try:
            query = session.query(ImageRecord)
            
            if exclude_videos:
                video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
                for ext in video_extensions:
                    query = query.filter(~ImageRecord.filename.ilike(f'%{ext}'))
            
            query = query.order_by(ImageRecord.created_at.desc())
            
            if offset is not None:
                query = query.offset(offset)
            if limit is not None:
                query = query.limit(limit)
            return query.all()
        finally:
            session.close()
    
    def get_image_by_id(self, image_id):
        session = self.SessionLocal()
        try:
            return session.query(ImageRecord).filter(ImageRecord.id == image_id).first()
        finally:
            session.close()

    def delete_image(self, image_id):
        session = self.SessionLocal()
        try:
            image = session.query(ImageRecord).filter(ImageRecord.id == image_id).first()
            if image:
                session.delete(image)
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            print(f"Delete failed: {e}")
            return False
        finally:
            session.close()

    def add_face_group(self, embedding):
        session = self.SessionLocal()
        try:
            embedding_json = json.dumps(embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding))
            group = FaceGroup(representative_embedding=embedding_json)
            session.add(group)
            session.commit()
            return group.id
        except Exception as e:
            session.rollback()
            print(f"Failed to add face group: {e}")
            return None
        finally:
            session.close()

    def add_detected_face(self, image_id, group_id, embedding, bbox, confidence):
        session = self.SessionLocal()
        try:
            embedding_json = json.dumps(embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding))
            bbox_json = json.dumps(bbox.tolist() if hasattr(bbox, 'tolist') else list(bbox))
            face = DetectedFace(
                image_id=image_id,
                group_id=group_id,
                embedding=embedding_json,
                bbox=bbox_json,
                confidence=float(confidence)
            )
            session.add(face)
            session.commit()
            return face.id
        except Exception as e:
            session.rollback()
            print(f"Failed to add detected face: {e}")
            return None
        finally:
            session.close()

    def get_all_face_groups(self):
        session = self.SessionLocal()
        try:
            return session.query(FaceGroup).all()
        finally:
            session.close()

    def get_faces_by_group(self, group_id):
        session = self.SessionLocal()
        try:
            return session.query(DetectedFace).filter(DetectedFace.group_id == group_id).all()
        finally:
            session.close()

    def get_face_by_id(self, face_id):
        session = self.SessionLocal()
        try:
            return session.query(DetectedFace).filter(DetectedFace.id == face_id).first()
        finally:
            session.close()
