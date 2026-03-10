# MLflow
Using mlflow for experimenting, tracking and logging models

### Commands
```bash
source .mlflow-lab-3.11/bin/activate
```

#### Run the model - each command in a new terminal
```bash
python3 train_model.py
```

Visit at http://127.0.0.1:5000
```bash
mlflow ui
mlflow models serve -m "models:/wine_quality/1" -p 5001 --env-manager=local
```
