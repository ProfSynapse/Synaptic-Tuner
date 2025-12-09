import os

ROOT_DIR = r"f:\Code\Toolset-Training\Datasets\tools_datasets\thinking"

def delete_backups():
    count = 0
    for root, dirs, files in os.walk(ROOT_DIR):
        for file in files:
            if file.endswith(".bak"):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    count += 1
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
    print(f"Deleted {count} backup files.")

if __name__ == "__main__":
    delete_backups()
