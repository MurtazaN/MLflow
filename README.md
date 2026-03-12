# MLflow Wine Quality Lab

Using MLflow for experimenting, tracking, and logging models.

## Setup

### 1. Activate the Virtual Environment

```bash
source .mlflow-lab-3.11/bin/activate
```

### 2. Install the Package (first time only)

```bash
pip install -e .
```

### 3. Run the Pipeline

```bash
./run.sh
```

This single command:
- Cleans old MLflow data (mlruns, mlflow.db, mlartifacts)
- Starts the MLflow UI in the background
- Loads and prepares wine quality data
- Trains a baseline Random Forest model
- Runs an XGBoost hyperparameter sweep (10 trials via Hyperopt)
- Registers and promotes the best model to Production
- Serves the production model and runs real-time inference

---

## Project Structure

```
MLflow/
├── src/
│   ├── __init__.py        # Package init
│   ├── data_prep.py       # Data loading and train/val/test splits
│   ├── train_model.py     # RF and XGBoost training functions
│   ├── run_pipeline.py    # Pipeline orchestration
│   └── inference.py       # Real-time inference client
├── data/                  # Wine quality CSV files
├── logs/                  # Pipeline, MLflow UI, and model server logs
├── run.sh                 # Entry point — orchestrates everything
├── pyproject.toml         # Setuptools config for editable install
└── starter.ipynb          # Original reference notebook
```

---

## Viewing Results

### MLflow UI — http://127.0.0.1:5000

1. **Runs** — `untuned_random_forest` (RF baseline), `xgboost_models` (click to see nested trials), `best_xgboost` (production model)
2. **Models** — 2 logged models: `random_forest_model` and `xgboost_model`
3. **Model Registry** — Click "Model registry" in sidebar to see version history and stage transitions

> **Tip:** To search for runs, use MLflow filter syntax:
> `tags.mlflow.runName = "xgboost_models"` (not free text)

---

## Manual Operation

### Serve the Model

```bash
source .mlflow-lab-3.11/bin/activate
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
mlflow models serve -m "models:/wine_quality/production" -p 5001 --env-manager=local
```

### Run Inference

```bash
python3 src/inference.py
```

### Test with curl

```bash
curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_split": {"columns": ["fixed_acidity","volatile_acidity","citric_acid","residual_sugar","chlorides","free_sulfur_dioxide","total_sulfur_dioxide","density","pH","sulphates","alcohol","is_red"], "data": [[7.0, 0.25, 0.36, 1.6, 0.034, 30.0, 110.0, 0.9906, 3.24, 0.50, 12.8, 0]]}}'
```

### Stop MLflow UI

The PID is printed at the end of `./run.sh`. Or:

```bash
kill $(lsof -t -i:5000)
```
