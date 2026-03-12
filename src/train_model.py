"""
Training module — trains Random Forest and XGBoost models, logs to MLflow.

Functions:
    train_rf()       — baseline Random Forest
    train_xgboost()  — hyperparameter sweep + retrain best model
"""

import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import mlflow.xgboost
import xgboost as xgb
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from mlflow.models.signature import infer_signature
from hyperopt import fmin, tpe, hp, Trials, SparkTrials, STATUS_OK
from hyperopt.pyll import scope


class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
    """Wraps sklearn classifier to return probabilities instead of binary predictions."""
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]


def train_rf(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Train a baseline Random Forest, log to MLflow, register the model,
    and print feature importance.

    Returns a dict with:
        model_name, rf_model, run_id, auc_score, model_version
    """
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
    model_version = mlflow.register_model(f"runs:/{run_id}/random_forest_model", model_name)
    time.sleep(5)

    # ── FEATURE IMPORTANCE ────────────────────────────────────
    print("\n" + "=" * 50)
    print("  FEATURE IMPORTANCE")
    print("=" * 50)
    feature_importances = pd.DataFrame(
        rf_model.feature_importances_,
        index=X_train.columns.tolist(),
        columns=['importance']
    )
    print(feature_importances.sort_values('importance', ascending=False))

    return {
        "model_name": model_name, "rf_model": rf_model,
        "run_id": run_id, "auc_score": auc_score,
        "model_version": model_version,
    }


def train_xgboost(X_train, X_val, X_test, y_train, y_val, y_test, model_name):
    """
    Run XGBoost hyperparameter sweep, retrain best model, and register it.

    Returns a dict with: model_version, auc
    """
    print("\n" + "=" * 50)
    print("  XGBOOST HYPERPARAMETER SWEEP (10 trials)")
    print("=" * 50)

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),
        'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
        'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
        'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
        'objective': 'binary:logistic',
        'seed': 123,
    }

    # Note: if using SparkTrials, copy data to local variables here to prevent
    # Spark workers from re-importing train_model:
    # _X_train, _X_val = X_train.copy(), X_val.copy()
    # _y_train, _y_val = y_train.copy(), y_val.copy()

    def train_xgb_model(params):
        """Train XGBoost model with given hyperparameters (used by Hyperopt)."""
        with mlflow.start_run(nested=True):
            train_dm = xgb.DMatrix(data=X_train, label=y_train)
            val_dm = xgb.DMatrix(data=X_val, label=y_val)
            booster = xgb.train(
                params=params, dtrain=train_dm, num_boost_round=1000,
                evals=[(val_dm, "validation")], early_stopping_rounds=50,
                verbose_eval=False
            )
            validation_predictions = booster.predict(val_dm)
            auc_score = roc_auc_score(y_val, validation_predictions)
            mlflow.log_metric('auc', auc_score)

            # Only log metrics per trial, NOT models.
            # We retrain the best config below and log that single model.
            return {'status': STATUS_OK, 'loss': -1 * auc_score, 'booster': booster.attributes()}

    # --- SparkTrials (for Databricks / parallel execution) ---
    # import pyspark
    # from pyspark import SparkContext, SparkConf
    # conf_spark = SparkConf().set("spark.driver.host", "127.0.0.1")
    # sc = SparkContext(conf=conf_spark)
    # spark_trials = SparkTrials(parallelism=10)

    with mlflow.start_run(run_name='xgboost_models'):
        best_params = fmin(
            fn=train_xgb_model,
            space=search_space,
            algo=tpe.suggest,
            max_evals=10,
            trials=Trials(),  # use spark_trials for parallel execution
        )

    # ── RETRAIN BEST MODEL ────────────────────────────────────
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['objective'] = 'binary:logistic'
    best_params['seed'] = 123

    print("\n" + "=" * 50)
    print("  BEST XGBOOST MODEL")
    print("=" * 50)

    with mlflow.start_run(run_name='best_xgboost') as best_run:
        train_dm = xgb.DMatrix(data=X_train, label=y_train)
        val_dm = xgb.DMatrix(data=X_val, label=y_val)
        test_dm = xgb.DMatrix(data=X_test, label=y_test)

        best_booster = xgb.train(
            params=best_params, dtrain=train_dm, num_boost_round=1000,
            evals=[(val_dm, "validation")], early_stopping_rounds=50,
            verbose_eval=False  # stops logs # TODO: Fix - so logs can be generated in a proper log file
        )

        test_predictions = best_booster.predict(test_dm)
        xgb_auc = roc_auc_score(y_test, test_predictions)
        mlflow.log_metric('auc', xgb_auc)

        signature = infer_signature(X_train, best_booster.predict(train_dm))
        mlflow.xgboost.log_model(best_booster, "xgboost_model", signature=signature)

        print(f"Best XGBoost AUC (test set): {xgb_auc}")

    # ── REGISTER MODEL ────────────────────────────────────────
    model_version = mlflow.register_model(
        f"runs:/{best_run.info.run_id}/xgboost_model", model_name
    )
    time.sleep(10)

    return {
        "model_version": model_version,
        "auc": xgb_auc,
    }


if __name__ == "__main__":
    from src.data_prep import load_data
    data = load_data()
    train_rf(**data)