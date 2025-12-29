import os
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build
import time

# Configuration
ACCOUNTS_DIR = Path("Service Account Utility/accounts")
SCOPES = ['https://www.googleapis.com/auth/drive']

def empty_drive():
    if not ACCOUNTS_DIR.exists():
        print(f"Error: Directory {ACCOUNTS_DIR} not found.")
        return

    accounts = sorted(list(ACCOUNTS_DIR.glob("*.json")))
    if not accounts:
        print(f"No service account files found in {ACCOUNTS_DIR}")
        return

    print(f"WARNING: This will delete ALL files from {len(accounts)} service accounts.")
    confirm = input("Are you sure you want to proceed? (type 'yes' to confirm): ")
    if confirm.lower() != 'yes':
        print("Aborted.")
        return

    for account_file in accounts:
        print(f"\nProcessing account: {account_file.name}")
        try:
            creds = service_account.Credentials.from_service_account_file(
                str(account_file), scopes=SCOPES)
            service = build('drive', 'v3', credentials=creds, cache_discovery=False)
            
            # List only files owned by the service account
            files_deleted = 0
            page_token = None
            while True:
                response = service.files().list(
                    q="'me' in owners and trashed = false",
                    spaces='drive',
                    fields='nextPageToken, files(id, name)',
                    pageToken=page_token
                ).execute()
                
                files = response.get('files', [])
                for file in files:
                    file_id = file.get('id')
                    file_name = file.get('name')
                    try:
                        print(f"  Deleting: {file_name} ({file_id})")
                        service.files().delete(fileId=file_id).execute()
                        files_deleted += 1
                    except Exception as delete_error:
                        print(f"  Permanent delete failed for {file_name}: {delete_error}")
                        if "insufficientFilePermissions" in str(delete_error) or "403" in str(delete_error):
                            try:
                                print(f"  Attempting to trash {file_name} instead...")
                                service.files().update(fileId=file_id, body={'trashed': True}).execute()
                                files_deleted += 1
                                print(f"  Successfully trashed {file_name}.")
                            except Exception as trash_error:
                                print(f"  Trash failed for {file_name}: {trash_error}")
                        else:
                            print(f"  Failed to delete {file_name}: {delete_error}")
                
                page_token = response.get('nextPageToken', None)
                if page_token is None:
                    break
            
            print(f"Finished {account_file.name}. Deleted {files_deleted} files.")
            
        except Exception as e:
            print(f"Error processing {account_file.name}: {e}")

    print("\nAll accounts processed.")

if __name__ == "__main__":
    empty_drive()
