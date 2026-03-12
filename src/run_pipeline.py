"""
Complete ML pipeline - extends train_model.py with XGBoost hyperparameter sweep,
model stage transitions, batch inference, and real-time inference.

Mirrors all functional steps from starter.ipynb.

Usage: ./run.sh  (or: python3 run_pipeline.py with mlflow ui running separately)
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
import mlflow.xgboost
import xgboost as xgb
import time
from sklearn.metrics import roc_auc_score
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from hyperopt import fmin, tpe, hp, Trials, SparkTrials, STATUS_OK
from hyperopt.pyll import scope
from pyspark.sql import SparkSession
from src.train_model import train


def main():
    # ============================================================
    # STEP 1: Run baseline Random Forest training & registration
    # ============================================================
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    result = train()
    X_train = result["X_train"]
    X_val = result["X_val"]
    X_test = result["X_test"]
    y_train = result["y_train"]
    y_val = result["y_val"]
    y_test = result["y_test"]
    model_name = result["model_name"]
    rf_model = result["rf_model"]
    run_id = result["run_id"]
    rf_auc = result["auc_score"]

    # ============================================================
    # STEP 2: Feature Importance Analysis
    # ============================================================
    print("\n" + "=" * 50)
    print("  FEATURE IMPORTANCE")
    print("=" * 50)
    feature_importances = pd.DataFrame(
        rf_model.feature_importances_,
        index=X_train.columns.tolist(),
        columns=['importance']
    )
    print(feature_importances.sort_values('importance', ascending=False))

    # ============================================================
    # STEP 3: Transition baseline model to Production
    # ============================================================
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{model_name}'")
    model_version = sorted(versions, key=lambda v: int(v.version))[-1]

    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Production",
    )
    print(f"\nModel version {model_version.version} transitioned to Production")

    # Sanity check: load production model and verify AUC
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/production")
    print(f"Production model AUC: {roc_auc_score(y_test, model.predict(X_test))}")

    # ============================================================
    # STEP 4: XGBoost Hyperparameter Sweep
    # ============================================================
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

    # Capture data in local variables so Spark workers don't re-import train_model
    _X_train, _X_val = X_train.copy(), X_val.copy()
    _y_train, _y_val = y_train.copy(), y_val.copy()

    def train_xgb_model(params):
        """Train XGBoost model with given hyperparameters (used by Hyperopt)."""
        with mlflow.start_run(nested=True):
            train_dm = xgb.DMatrix(data=_X_train, label=_y_train)
            val_dm = xgb.DMatrix(data=_X_val, label=_y_val)
            booster = xgb.train(
                params=params, dtrain=train_dm, num_boost_round=1000,
                evals=[(val_dm, "validation")], early_stopping_rounds=50,
                verbose_eval=False
            )
            validation_predictions = booster.predict(val_dm)
            auc_score = roc_auc_score(_y_val, validation_predictions)
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

    # ============================================================
    # STEP 5: Retrain best XGBoost model and log it
    # ============================================================
    # best_params from fmin has raw values; reconstruct full params
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
            verbose_eval=False # stops logs # TODO: Fix - so logs can be generated in a proper log file
        )

        test_predictions = best_booster.predict(test_dm)
        xgb_auc = roc_auc_score(y_test, test_predictions)
        mlflow.log_metric('auc', xgb_auc)

        signature = infer_signature(X_train, best_booster.predict(train_dm))
        mlflow.xgboost.log_model(best_booster, "xgboost_model", signature=signature)

        print(f"Best XGBoost AUC (test set): {xgb_auc}")

    # ============================================================
    # STEP 6: Update production model with best XGBoost model
    # ============================================================
    new_model_version = mlflow.register_model(
        f"runs:/{best_run.info.run_id}/xgboost_model", model_name
    )
    time.sleep(10)

    # Archive old version, promote new version
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage='Archived'
    )
    client.transition_model_version_stage(
        name=model_name,
        version=new_model_version.version,
        stage='Production'
    )

    print(f"Model version {new_model_version.version} promoted to Production")
    print(f"Model version {model_version.version} archived")

    # ============================================================
    # STEP 7: Batch Inference with Spark
    # ============================================================
    # SparkSession imported at top of file

    spark = SparkSession.builder \
        .appName("MLflow Integration") \
        .config("spark.some.config.option", "config-value") \
        .getOrCreate()

    apply_model_udf = mlflow.pyfunc.spark_udf(spark, f"models:/{model_name}/production")

    # To run batch inference on new data, uncomment and set table_path:
    # new_data = spark.read.format("csv").load(table_path)
    # predictions = new_data.withColumn("prediction", apply_model_udf(*new_data.columns))
    # predictions.show()

    # ============================================================
    # RESULTS SUMMARY
    # ============================================================
    print("\n" + "=" * 50)
    print("  RESULTS SUMMARY")
    print("=" * 50)
    print(f"Model name:              {model_name}")
    print(f"RF baseline AUC:         {rf_auc}")
    print(f"XGBoost best AUC:        {xgb_auc}")
    print(f"Production model version: {new_model_version.version}")
    print(f"\nView results: http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
