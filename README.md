# MLflow Wine Quality Lab

Using MLflow for experimenting, tracking, and logging models.

## Setup

### 1. Activate the Virtual Environment

```bash
source .mlflow-lab-3.11/bin/activate
```

### 2. Run the Pipeline

```bash
./run.sh
```

This single command does everything:
- Cleans old MLflow data
- Starts the MLflow UI in the background
- Trains a baseline Random Forest model
- Runs an XGBoost hyperparameter sweep (96 trials via Hyperopt + SparkTrials)
- Registers and promotes the best model to Production
- Serves the production model and runs real-time inference
- Prints all results to your terminal

> **Requires:** `pyspark` installed in your venv (`pip install pyspark`)

---

## Viewing Results

### MLflow UI — http://127.0.0.1:5000

All ML results (experiments, metrics, models) are here:

1. **Experiments → Runs** — Click on runs to see parameters, metrics, and artifacts
2. **"untuned_random_forest"** — The baseline Random Forest run
3. **"xgboost_models"** — Click this parent run, then expand to see ~96 nested child runs (one per hyperparameter trial)
4. **Model Registry** — Click "Models" in the sidebar to see registered model versions and stage transitions

> **Tip:** To search for runs, use MLflow filter syntax:
> `tags.mlflow.runName = "xgboost_models"` (not free text)

### Spark UI — http://127.0.0.1:4040

Spark job monitoring (only available while the pipeline is actively running). Shows task progress and DAG visualizations.

---

## Project Files

| File              | Purpose                                                    |
|-------------------|------------------------------------------------------------|
| `train_model.py`  | Baseline Random Forest training & registration             |
| `run_pipeline.py` | Full pipeline: RF → XGBoost sweep → model promotion        |
| `inference.py`    | Real-time inference against the served production model    |
| `run.sh`          | Entry point — orchestrates everything                      |
| `pipeline.log`    | Spark/MLflow/XGBoost warnings from the pipeline            |
| `mlflow_ui.log`   | MLflow UI server output                                    |
| `model_server.log`| MLflow model server output                                 |

---

## Manual Operation (if needed)

### Serve the Model

In a new terminal:

```bash
source .mlflow-lab-3.11/bin/activate
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
mlflow models serve -m "models:/wine_quality/production" -p 5001 --env-manager=local
```

### Test Predictions

```bash
curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_split": {"columns": ["fixed_acidity","volatile_acidity","citric_acid","residual_sugar","chlorides","free_sulfur_dioxide","total_sulfur_dioxide","density","pH","sulphates","alcohol","is_red"], "data": [[7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 1]]}}'
```

### Stop MLflow UI

The PID is printed at the end of `./run.sh`. Or:

```bash
kill $(lsof -t -i:5000)
```
