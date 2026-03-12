#!/bin/bash
# Single entry point for the MLflow Wine Quality pipeline.

set -e

echo "=========================================="
echo "  MLflow Wine Quality Pipeline"
echo "=========================================="

# 1. Clean old MLflow data
echo "Cleaning old MLflow data..."
rm -rf mlruns mlflow.db
rm -rf mlartifacts
mkdir -p logs

# 2. Start MLflow UI in background
echo "Starting MLflow UI (background)..."
mlflow ui > logs/mlflow_ui.log 2>&1 &
MLFLOW_PID=$!
sleep 3
echo "MLflow UI running at http://127.0.0.1:5000 (PID: $MLFLOW_PID)"

# 3. Run the pipeline
echo ""
echo "Running pipeline..."
echo "=========================================="
python3 src/run_pipeline.py 2>logs/pipeline.log

# 4. Serve the production model and run real-time inference
echo ""
echo "Starting model server on port 5001..."
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
mlflow models serve -m "models:/wine_quality/production" -p 5001 --env-manager=local > logs/model_server.log 2>&1 &
MODEL_PID=$!
sleep 10

echo "Running real-time inference..."
python3 src/inference.py

# 5. Cleanup
kill $MODEL_PID 2>/dev/null || true

echo ""
echo "=========================================="
echo "Pipeline finished successfully!"
echo ""
echo "MLflow UI: http://127.0.0.1:5000"
echo "Logs:      logs/pipeline.log, logs/mlflow_ui.log"
echo "To stop MLflow UI: kill $MLFLOW_PID"
echo "=========================================="
