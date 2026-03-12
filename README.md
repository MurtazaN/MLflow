# MLflow Wine Quality Lab

Using MLflow for experimenting, tracking, and logging models.

## Setup (Local)

### 1. Create and Activate Virtual Environment

```bash
python3.11 -m venv .mlflow-lab-3.11
source .mlflow-lab-3.11/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
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

### 4. View Results

- **MLflow UI** — http://127.0.0.1:5000
- **Runs** — `untuned_random_forest`, `xgboost_models` (expand for nested trials), `best_xgboost`
- **Model Registry** — Click "Model registry" in sidebar for version history

> **Tip:** Filter runs with MLflow syntax: `tags.mlflow.runName = "xgboost_models"`

### 5. Stop MLflow UI

```bash
kill $(lsof -t -i:5000)
```

---

## Setup (Docker)

See [DOCKER_SETUP.md](DOCKER_SETUP.md) for full instructions on running with Docker Compose.

---

## Project Structure

```
MLflow/
├── src/
│   ├── __init__.py          # Package init
│   ├── data_prep.py         # Data loading and train/val/test splits
│   ├── train_model.py       # RF and XGBoost training functions
│   ├── run_pipeline.py      # Pipeline orchestration
│   ├── inference.py         # CLI inference client
│   └── fastapi-app.py       # FastAPI inference service (Docker)
├── data/                    # Wine quality CSV files
├── logs/                    # Pipeline, MLflow UI, and model server logs
├── run.sh                   # Local entry point
├── docker-compose.yml       # Docker Compose orchestration
├── Dockerfile               # Container image definition
├── .env                     # Environment variables (not tracked in git)
├── pyproject.toml           # Setuptools config for editable install
├── requirements.txt         # Python dependencies
└── starter.ipynb            # Original reference notebook
```

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
