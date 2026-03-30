# =========================
# IMPORTS
# =========================
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# =========================
# HUGGING FACE SETUP
# =========================
# Use token from environment (GitHub Actions / local)
api = HfApi(token=os.getenv("HF_TOKEN"))

# =========================
# LOAD DATASET FROM HF
# =========================
# Dataset must exist on HF Dataset Hub
DATASET_PATH = "hf://datasets/shreyackdeshpande/tourism/tourism.csv"

df = pd.read_csv(DATASET_PATH)
print("✅ Dataset loaded successfully.")

# =========================
# DATA CLEANING
# =========================

# 1. Drop unnecessary columns (IDs, index columns)
drop_cols = ["Unnamed: 0", "CustomerID"]
df.drop(columns=drop_cols, inplace=True, errors="ignore")

# 2. Handle missing values

# Numerical columns → fill with median
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Categorical columns → fill with mode
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    # Avoid chained assignment warning
    df[col] = df[col].fillna(df[col].mode()[0])

print(f"✅ Data cleaned. Shape: {df.shape}")

# =========================
# DEFINE TARGET
# =========================
target_col = "ProdTaken"

# Split features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# =========================
# TRAIN-TEST SPLIT
# =========================
# Using stratify to maintain class balance
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("✅ Train-test split completed.")

# =========================
# SAVE FILES LOCALLY
# =========================
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("✅ Files saved locally.")

# =========================
# CREATE DATASET REPO (IF NOT EXISTS)
# =========================
# Ensures pipeline doesn't fail in CI/CD
api.create_repo(
    repo_id="shreyackdeshpande/tourism",
    repo_type="dataset",
    exist_ok=True
)

# =========================
# UPLOAD FILES TO HF DATASET HUB
# =========================
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    try:
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file_path,
            repo_id="shreyackdeshpande/tourism",
            repo_type="dataset"
        )
        print(f"✅ Uploaded: {file_path}")
    except Exception as e:
        print(f"❌ Failed to upload {file_path}: {e}")

print("🚀 All dataset files uploaded successfully!")
