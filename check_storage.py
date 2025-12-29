import os
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Configuration
ACCOUNTS_DIR = Path("Service Account Utility/accounts")
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def format_size(size_bytes):
    """Formats bytes into human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def check_storage():
    if not ACCOUNTS_DIR.exists():
        print(f"Error: Directory {ACCOUNTS_DIR} not found.")
        return

    accounts = sorted(list(ACCOUNTS_DIR.glob("*.json")))
    if not accounts:
        print(f"No service account files found in {ACCOUNTS_DIR}")
        return

    print(f"Checking storage for {len(accounts)} service accounts...\n")
    print(f"{'Account':<15} | {'Usage':<15} | {'Limit':<15} | {'% Used':<10}")
    print("-" * 60)

    total_usage = 0
    total_limit = 0

    for account_file in accounts:
        try:
            creds = service_account.Credentials.from_service_account_file(
                str(account_file), scopes=SCOPES)
            service = build('drive', 'v3', credentials=creds, cache_discovery=False)
            
            about = service.about().get(fields="storageQuota").execute()
            quota = about.get('storageQuota', {})
            limit = int(quota.get('limit', 0))
            usage = int(quota.get('usage', 0))
            
            percent = (usage / limit * 100) if limit > 0 else 0
            
            print(f"{account_file.name:<15} | {format_size(usage):<15} | {format_size(limit):<15} | {percent:>6.2f}%")
            
            total_usage += usage
            total_limit += limit
            
        except Exception as e:
            print(f"{account_file.name:<15} | Error: {e}")

    print("-" * 60)
    total_percent = (total_usage / total_limit * 100) if total_limit > 0 else 0
    print(f"{'TOTAL':<15} | {format_size(total_usage):<15} | {format_size(total_limit):<15} | {total_percent:>6.2f}%")

if __name__ == "__main__":
    check_storage()
