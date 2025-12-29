from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
import io
import os
import json
import time
from pathlib import Path

import threading

SCOPES = ['https://www.googleapis.com/auth/drive']

class DriveClient:
    def __init__(self, service_account_path):
        self.service_account_path = Path(service_account_path)
        self.accounts = []
        self.current_account_index = 0
        self._thread_local = threading.local()
        
        if self.service_account_path.is_dir():
            # Load all JSON files from directory
            self.accounts = sorted(list(self.service_account_path.glob("*.json")))
            if not self.accounts:
                raise Exception(f"No service account files found in {service_account_path}")
            print(f"Found {len(self.accounts)} service accounts.")
        else:
            # Single file
            self.accounts = [self.service_account_path]

    @property
    def service(self):
        """Get or create a thread-local service object."""
        if not hasattr(self._thread_local, 'service'):
            self._load_account(self.current_account_index)
        return self._thread_local.service

    def _load_account(self, index):
        """Load service account at specific index for the current thread."""
        if index >= len(self.accounts):
            index = 0 # Loop back to start
        
        self.current_account_index = index
        account_file = self.accounts[index]
        print(f"Thread {threading.get_ident()} switching to service account: {account_file.name}")
        
        creds = service_account.Credentials.from_service_account_file(
            str(account_file), scopes=SCOPES)
        self._thread_local.service = build('drive', 'v3', credentials=creds, cache_discovery=False)

    def _rotate_account(self):
        """Switch to the next available service account for the current thread."""
        next_index = (self.current_account_index + 1) % len(self.accounts)
        self._load_account(next_index)

    def get_storage_quota(self):
        """Get storage quota usage for current account."""
        try:
            about = self.service.about().get(fields="storageQuota").execute()
            quota = about.get('storageQuota', {})
            limit = int(quota.get('limit', 0))
            usage = int(quota.get('usage', 0))
            return usage, limit
        except Exception as e:
            print(f"Failed to get quota: {e}")
            return 0, 0

    def list_images_in_folder(self, folder_id):
        """List all image files in the specified folder."""
        results = []
        page_token = None
        while True:
            try:
                # Query for files in folder and is an image
                query = f"'{folder_id}' in parents and mimeType contains 'image/' and trashed = false"
                response = self.service.files().list(
                    q=query,
                    spaces='drive',
                    fields='nextPageToken, files(id, name, mimeType, webContentLink, webViewLink)',
                    pageToken=page_token
                ).execute()
                
                results.extend(response.get('files', []))
                page_token = response.get('nextPageToken', None)
                if page_token is None:
                    break
            except Exception as e:
                print(f"List failed with current account: {e}")
                # Try rotating if it's a quota issue or permission issue
                # For now, just re-raise to avoid infinite loops in listing
                raise e
        return results

    def download_file(self, file_id):
        """Download file content as bytes with retries."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                request = self.service.files().get_media(fileId=file_id)
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request, chunksize=1024*1024) # 1MB chunks
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
                return fh.getvalue()
            except Exception as e:
                print(f"Download attempt {attempt + 1} failed for {file_id}: {e}")
                if attempt == max_retries - 1:
                    raise e
                time.sleep(1) # Wait before retry

    def upload_file(self, filename, file_content, folder_id, mime_type='image/jpeg'):
        """Upload a file to Google Drive with rotation support and max 2 retries."""
        max_attempts = 3
        attempts = 0
        
        while attempts < max_attempts:
            try:
                # Check quota first
                usage, limit = self.get_storage_quota()
                if limit > 0 and usage >= (limit - 10 * 1024 * 1024): # Leave 10MB buffer
                    print(f"Account {self.accounts[self.current_account_index].name} is full. Rotating...")
                    self._rotate_account()
                    attempts += 1
                    continue

                file_metadata = {
                    'name': filename,
                    'parents': [folder_id]
                }
                media = MediaIoBaseUpload(io.BytesIO(file_content), mimetype=mime_type, resumable=True)
                file = self.service.files().create(body=file_metadata, media_body=media, fields='id').execute()
                return file.get('id')
            except Exception as e:
                attempts += 1
                print(f"Upload attempt {attempts} failed with account {self.accounts[self.current_account_index].name}: {e}")
                
                if attempts < max_attempts:
                    if "storageQuotaExceeded" in str(e):
                        print("Quota exceeded. Rotating account and retrying...")
                        self._rotate_account()
                    else:
                        print("Retrying with next account...")
                        self._rotate_account()
                    time.sleep(1)
                else:
                    print(f"All {max_attempts} upload attempts failed.")
        
        return None

    def get_file_metadata(self, file_id):
        return self.service.files().get(fileId=file_id, fields='id, name, mimeType').execute()

    def delete_file(self, file_id):
        """Delete a file from Google Drive (fallback to trash)."""
        try:
            # Try permanent delete first
            self.service.files().delete(fileId=file_id).execute()
            return True
        except Exception as e:
            print(f"Permanent delete failed for {file_id}: {e}")
            if "insufficientFilePermissions" in str(e) or "403" in str(e):
                try:
                    # Fallback to trash
                    print(f"Attempting to trash file {file_id} instead...")
                    self.service.files().update(fileId=file_id, body={'trashed': True}).execute()
                    return True
                except Exception as trash_error:
                    print(f"Trash failed for {file_id}: {trash_error}")
                    return False
            return False
