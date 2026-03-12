import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from mlflow.models.signature import infer_signature
import time


class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
    """Wraps sklearn classifier to return probabilities instead of binary predictions."""
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]


def train():
    """
    Train a baseline Random Forest, log to MLflow, and register the model.

    Returns a dict with all variables the pipeline needs:
        X_train, X_val, X_test, y_train, y_val, y_test,
        model_name, rf_model, run_id, auc_score
    """
    # Point MLflow to the local tracking server
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # ── DATA PREPARATION ──────────────────────────────────────
    print("Loading data...")
    white_wine = pd.read_csv("data/winequality-white.csv", sep=";")
    red_wine = pd.read_csv("data/winequality-red.csv", sep=",")

    red_wine['is_red'] = 1
    white_wine['is_red'] = 0
    data = pd.concat([red_wine, white_wine], axis=0)
    data.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

    # Define high quality as >= 7
    data['quality'] = (data.quality >= 7).astype(int)

    X = data.drop(["quality"], axis=1)
    y = data.quality

    X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.6, random_state=123)
    X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=123)

    # ── TRAINING RUN ──────────────────────────────────────────
    model_name = "wine_quality"

    with mlflow.start_run(run_name='untuned_random_forest') as run:
        n_estimators = 10
        rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=123)
        rf_model.fit(X_train, y_train)

        predictions = rf_model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, predictions)

        mlflow.log_param('n_estimators', n_estimators)
        mlflow.log_metric('auc', auc_score)

        wrappedModel = SklearnModelWrapper(rf_model)
        signature = infer_signature(X_train, wrappedModel.predict(None, X_train))

        mlflow.pyfunc.log_model(
            "random_forest_model",
            python_model=wrappedModel,
            signature=signature,
        )

        print(f"Random Forest AUC: {auc_score}")
        run_id = run.info.run_id

    # ── REGISTER MODEL ────────────────────────────────────────
    print(f"Registering model: {model_name}...")
    mlflow.register_model(f"runs:/{run_id}/random_forest_model", model_name)
    time.sleep(5)  # wait for registration to complete

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "model_name": model_name, "rf_model": rf_model,
        "run_id": run_id, "auc_score": auc_score,
    }


if __name__ == "__main__":
    train()