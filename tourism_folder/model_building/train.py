# imports
import pandas as pd
import os
import joblib
import mlflow
import xgboost as xgb

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError


#mlflow setup
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-experiment")


# loading dataset
DATA_PATH = "hf://datasets/shreyackdeshpande/tourism/tourism.csv"

df = pd.read_csv(DATA_PATH)
print("Dataset loaded successfully!")

# Drop unnecessary columns
df.drop(columns=["Unnamed: 0", "CustomerID"], inplace=True, errors="ignore")

# Handle missing values
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

# Remove target from num_cols if present
target_col = "ProdTaken"
if target_col in num_cols:
    num_cols.remove(target_col)

# Fill missing values
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])


# split date
X = df.drop(columns=[target_col])
y = df[target_col]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# handle class imbalance
class_counts = ytrain.value_counts()
class_weight = class_counts.get(0, 1) / class_counts.get(1, 1)


preprocessor = make_column_transformer(
    (StandardScaler(), num_cols),
    (OneHotEncoder(handle_unknown='ignore'), cat_cols)
)

#model
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)


#hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 75],
    'xgbclassifier__max_depth': [3, 4],
    'xgbclassifier__learning_rate': [0.05, 0.1]
}


#pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)


#training andn mlflow
with mlflow.start_run():

    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(Xtrain, ytrain)

    best_model = grid_search.best_estimator_

    # Predictions
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log parameters
    mlflow.log_params(grid_search.best_params_)

    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "test_accuracy": test_report["accuracy"]
    })

    # Save model
    model_path = "best_tourism_model_v1.joblib"
    joblib.dump(best_model, model_path)

    mlflow.log_artifact(model_path)

    print("✅ Model trained and saved!")


#upload 
api = HfApi(token=os.getenv("HF_TOKEN"))

repo_id = "shreyackdeshpande/tourism-model"
repo_type = "model"

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Model repo '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"Creating model repo '{repo_id}'...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)

# Upload model
api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo=model_path,
    repo_id=repo_id,
    repo_type=repo_type,
)

print("🚀 Model uploaded to Hugging Face!")
