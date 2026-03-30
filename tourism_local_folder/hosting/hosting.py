# =========================
# IMPORTS
# =========================
from huggingface_hub import HfApi, create_repo
import os
import time

# =========================
# CONFIG
# =========================
repo_id = "shreyackdeshpande/tourism-space-HF"
repo_type = "space"

api = HfApi(token=os.getenv("HF_TOKEN"))

# =========================
# CREATE SPACE IF NOT EXISTS
# =========================
create_repo(
    repo_id=repo_id,
    repo_type=repo_type,
    space_sdk="docker",
    exist_ok=True
)

print("✅ Space ready.")

# =========================
# UPLOAD DEPLOYMENT FILES
# =========================
for i in range(3):
    try:
        api.upload_folder(
            folder_path="tourism_local_folder/deployment",
            repo_id=repo_id,
            repo_type=repo_type,
            path_in_repo=""
        )
        print("🚀 Deployment successful!")
        break
    except Exception as e:
        print(f"Retry {i+1} failed:", e)
        time.sleep(5)
