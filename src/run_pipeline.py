"""
Pipeline orchestration — loads data, calls training functions, manages model lifecycle.

Usage: ./run.sh  (or: python3 src/run_pipeline.py with mlflow ui running separately)
"""

import os
import mlflow
from sklearn.metrics import roc_auc_score
from mlflow.tracking import MlflowClient
# from pyspark.sql import SparkSession
from src.data_prep import load_data
from src.train_model import train_rf, train_xgboost


def main():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # ============================================================
    # STEP 1: Load and prepare data
    # ============================================================
    data = load_data()

    # ============================================================
    # STEP 2: Train baseline Random Forest
    # ============================================================
    rf = train_rf(**data)

    # ============================================================
    # STEP 3: Promote RF to Production
    # ============================================================
    client.transition_model_version_stage(
        name=rf["model_name"],
        version=rf["model_version"].version,
        stage="Production",
    )
    print(f"\nModel version {rf['model_version'].version} transitioned to Production")

    # Sanity check: load production model and verify AUC
    model = mlflow.pyfunc.load_model(f"models:/{rf['model_name']}/production")
    print(f"Production model AUC: {roc_auc_score(data['y_test'], model.predict(data['X_test']))}")

    # ============================================================
    # STEP 4: Train XGBoost + register best model
    # ============================================================
    xgb_result = train_xgboost(**data, model_name=rf["model_name"])

    # ============================================================
    # STEP 5: Promote XGBoost, archive RF
    # ============================================================
    client.transition_model_version_stage(
        name=rf["model_name"],
        version=rf["model_version"].version,
        stage="Archived",
    )
    client.transition_model_version_stage(
        name=rf["model_name"],
        version=xgb_result["model_version"].version,
        stage="Production",
    )
    print(f"Model version {xgb_result['model_version'].version} promoted to Production")
    print(f"Model version {rf['model_version'].version} archived")

    # ============================================================
    # STEP 6: Batch Inference with Spark (placeholder)
    # ============================================================
    # spark = SparkSession.builder \
    #     .appName("MLflow Integration") \
    #     .config("spark.some.config.option", "config-value") \
    #     .getOrCreate()
    #
    # apply_model_udf = mlflow.pyfunc.spark_udf(spark, f"models:/{rf['model_name']}/production")
    #
    # new_data = spark.read.format("csv").load(table_path)
    # predictions = new_data.withColumn("prediction", apply_model_udf(*new_data.columns))
    # predictions.show()

    # ============================================================
    # RESULTS SUMMARY
    # ============================================================
    print("\n" + "=" * 50)
    print("  RESULTS SUMMARY")
    print("=" * 50)
    print(f"Model name:              {rf['model_name']}")
    print(f"RF baseline AUC:         {rf['auc_score']}")
    print(f"XGBoost best AUC:        {xgb_result['auc']}")
    print(f"Production model version: {xgb_result['model_version'].version}")
    print(f"\nView results: http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
