from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import json
from datetime import datetime
from config import DATABASE_URL
from models import Base, ImageRecord, FaceGroup, DetectedFace, Folder

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
                created_at=datetime.now(),
                folder_id=kwargs.get('folder_id')
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
    
    def get_all_images(self, limit=None, offset=None, exclude_videos=True, folder_id=None):
        session = self.SessionLocal()
        try:
            query = session.query(ImageRecord)
            
            if exclude_videos:
                video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
                for ext in video_extensions:
                    query = query.filter(~ImageRecord.filename.ilike(f'%{ext}'))
            
            if folder_id is not None:
                query = query.filter(ImageRecord.folder_id == folder_id)
            
            query = query.order_by(ImageRecord.created_at.desc())
            
            if offset is not None:
                query = query.offset(offset)
            if limit is not None:
                query = query.limit(limit)
            return query.all()
        finally:
            session.close()
    
    def count_vectors(self, folder_id=None):
        session = self.SessionLocal()
        try:
            query = session.query(ImageRecord).filter(ImageRecord.embedding != None)
            if folder_id is not None:
                query = query.filter(ImageRecord.folder_id == folder_id)
            return query.count()
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

    def get_all_face_groups(self, folder_id=None):
        session = self.SessionLocal()
        try:
            if folder_id is None:
                return session.query(FaceGroup).all()
            else:
                # Get groups that have faces in images belonging to the folder
                return session.query(FaceGroup).join(DetectedFace).join(ImageRecord).filter(ImageRecord.folder_id == folder_id).distinct().all()
        finally:
            session.close()

    def get_faces_by_group(self, group_id, folder_id=None):
        session = self.SessionLocal()
        try:
            query = session.query(DetectedFace).filter(DetectedFace.group_id == group_id)
            if folder_id is not None:
                query = query.join(ImageRecord).filter(ImageRecord.folder_id == folder_id)
            return query.all()
        finally:
            session.close()

    def get_face_by_id(self, face_id):
        session = self.SessionLocal()
        try:
            return session.query(DetectedFace).filter(DetectedFace.id == face_id).first()
        finally:
            session.close()

    # Folder methods
    def add_folder(self, name):
        session = self.SessionLocal()
        try:
            folder = Folder(name=name)
            session.add(folder)
            session.commit()
            return folder.id
        except Exception as e:
            session.rollback()
            print(f"Failed to add folder: {e}")
            return None
        finally:
            session.close()

    def get_all_folders(self):
        session = self.SessionLocal()
        try:
            return session.query(Folder).all()
        finally:
            session.close()

    def get_folder_by_id(self, folder_id):
        session = self.SessionLocal()
        try:
            return session.query(Folder).filter(Folder.id == folder_id).first()
        finally:
            session.close()

    def get_folder_by_name(self, name):
        session = self.SessionLocal()
        try:
            return session.query(Folder).filter(Folder.name == name).first()
        finally:
            session.close()

    def delete_folder(self, folder_id):
        session = self.SessionLocal()
        try:
            folder = session.query(Folder).filter(Folder.id == folder_id).first()
            if folder:
                # Set folder_id to null for images in this folder
                session.query(ImageRecord).filter(ImageRecord.folder_id == folder_id).update({ImageRecord.folder_id: None})
                session.delete(folder)
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            print(f"Failed to delete folder: {e}")
            return False
        finally:
            session.close()
