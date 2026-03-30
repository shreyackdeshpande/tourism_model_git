# =========================
# IMPORTS
# =========================
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import xgboost as xgb
import joblib
import os
import mlflow
import time
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# =========================
# MLFLOW SETUP (CI/CD SAFE)
# =========================
# Use local file-based tracking (works in GitHub Actions)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("tourism-training-experiment")

# =========================
# LOAD DATA FROM HF DATASET
# =========================
Xtrain = pd.read_csv("hf://datasets/shreyackdeshpande/tourism/Xtrain.csv")
Xtest = pd.read_csv("hf://datasets/shreyackdeshpande/tourism/Xtest.csv")
ytrain = pd.read_csv("hf://datasets/shreyackdeshpande/tourism/ytrain.csv")
ytest = pd.read_csv("hf://datasets/shreyackdeshpande/tourism/ytest.csv")

print("✅ Dataset loaded successfully.")

# =========================
# FIX TARGET SHAPE
# =========================
ytrain = ytrain.squeeze()
ytest = ytest.squeeze()

# =========================
# AUTO DETECT FEATURE TYPES (ROBUST)
# =========================
num_cols = Xtrain.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = Xtrain.select_dtypes(include=['object']).columns.tolist()

# =========================
# CLASS WEIGHT (SAFE)
# =========================
class_counts = ytrain.value_counts()

class_weight = (
    class_counts.get(0, 1) / class_counts.get(1, 1)
    if class_counts.get(1, 0) != 0 else 1
)

# =========================
# PREPROCESSOR (ONLY USED IF RAW DATA)
# =========================
preprocessor = make_column_transformer(
    (StandardScaler(), num_cols),
    (OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
)

# =========================
# MODEL
# =========================
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42
)

# =========================
# AUTO DETECT IF DATA IS ALREADY ENCODED
# =========================
missing_cols = [col for col in cat_cols if col not in Xtrain.columns]

if missing_cols:
    print("⚠️ Detected PREPROCESSED data → Skipping encoder")
    model_pipeline = xgb_model
    param_grid = {
        'n_estimators': [50, 75],
        'max_depth': [3, 4],
        'learning_rate': [0.05, 0.1],
    }
else:
    print("✅ Detected RAW data → Using full pipeline")
    model_pipeline = make_pipeline(preprocessor, xgb_model)
    param_grid = {
        'xgbclassifier__n_estimators': [50, 75],
        'xgbclassifier__max_depth': [3, 4],
        'xgbclassifier__learning_rate': [0.05, 0.1],
    }

# =========================
# TRAINING WITH MLFLOW
# =========================
with mlflow.start_run():

    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    mlflow.log_params(grid_search.best_params_)

    best_model = grid_search.best_estimator_

    # =========================
    # PREDICTIONS
    # =========================
    threshold = 0.45

    y_pred_train = (best_model.predict_proba(Xtrain)[:, 1] >= threshold).astype(int)
    y_pred_test = (best_model.predict_proba(Xtest)[:, 1] >= threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # =========================
    # LOG METRICS (SAFE)
    # =========================
    mlflow.log_metrics({
        "train_accuracy": train_report.get('accuracy', 0),
        "train_precision": train_report.get('1', {}).get('precision', 0),
        "train_recall": train_report.get('1', {}).get('recall', 0),
        "train_f1": train_report.get('1', {}).get('f1-score', 0),
        "test_accuracy": test_report.get('accuracy', 0),
        "test_precision": test_report.get('1', {}).get('precision', 0),
        "test_recall": test_report.get('1', {}).get('recall', 0),
        "test_f1": test_report.get('1', {}).get('f1-score', 0),
    })

    # =========================
    # SAVE MODEL
    # =========================
    model_path = "best_tourism_model.joblib"
    joblib.dump(best_model, model_path)

    mlflow.log_artifact(model_path)

    print("✅ Model saved successfully.")

    # =========================
    # UPLOAD TO HF MODEL HUB
    # =========================
    api = HfApi(token=os.getenv("HF_TOKEN"))
    repo_id = "shreyackdeshpande/tourism-model"

    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
        print("Model repo exists.")
    except RepositoryNotFoundError:
        print("Creating model repo...")
        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    # Retry logic for upload (handles HF failures)
    for i in range(3):
        try:
            api.upload_file(
                path_or_fileobj=model_path,
                path_in_repo=model_path,
                repo_id=repo_id,
                repo_type="model"
            )
            print("🚀 Model uploaded successfully!")
            break
        except Exception as e:
            print(f"Retry {i+1} failed:", e)
            time.sleep(5)
