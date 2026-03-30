# =========================
# IMPORTS
# =========================
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os
import time

# =========================
# CONFIG
# =========================
repo_id = "shreyackdeshpande/tourism"
repo_type = "dataset"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# =========================
# CHECK / CREATE DATASET REPO
# =========================
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"Dataset '{repo_id}' not found. Creating new dataset...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset '{repo_id}' created.")

# =========================
# UPLOAD DATA FOLDER
# =========================
for i in range(3):
    try:
        api.upload_folder(
            folder_path="tourism_local_folder/data",
            repo_id=repo_id,
            repo_type=repo_type,
        )
        print("🚀 Dataset uploaded successfully!")
        break
    except Exception as e:
        print(f"Retry {i+1} failed:", e)
        time.sleep(5)
