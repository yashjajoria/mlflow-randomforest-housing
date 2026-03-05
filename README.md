# MLflow RandomForest Housing Price Prediction

A portfolio-friendly **Machine Learning / MLOps** project that demonstrates **experiment tracking** and **model versioning** with **MLflow**. The project trains a **`RandomForestRegressor`** on the **California Housing** dataset and performs **hyperparameter tuning** via **`GridSearchCV`**.

---

## Project Overview
This repository showcases an end-to-end workflow that is common in production ML systems:

- Load a canonical benchmark dataset (**California Housing**, `sklearn.datasets`).
- Train a **RandomForest** regression model.
- Tune hyperparameters using **cross-validated grid search**.
- Track experiments with **MLflow Tracking** (params, metrics, artifacts).
- Infer and log the **model signature** for safe deployment/serving.
- Register model versions in the **MLflow Model Registry**.
- Reload the registered model and run basic **validation**.

The goal is to provide a clean reference implementation for MLflow-based experiment management and model lifecycle practices.

---

## Tech Stack
- **Python** (ML development)
- **scikit-learn** (dataset, model training, GridSearchCV)
- **pandas / numpy** (data handling)
- **MLflow** (tracking, artifacts, model registry)

---

## Project Architecture
A typical run of this project follows the steps below:

1. **Data Loading**
   - Load California Housing dataset from `sklearn`.
   - Split into train/test sets.

2. **Training + Hyperparameter Tuning**
   - Train `RandomForestRegressor`.
   - Run `GridSearchCV` for best hyperparameters.

3. **MLflow Experiment Tracking**
   - Log:
     - Parameters (e.g., `n_estimators`, `max_depth`, etc.)
     - Metrics (e.g., RMSE, MAE, R²)
     - Model artifact
     - Signature (inputs/outputs)

4. **Model Registry**
   - Register best model as a versioned MLflow model.

5. **Model Loading + Validation**
   - Load the model from MLflow.
   - Run predictions on a validation/test split to verify correctness.

> Note: File/module names can vary; the architecture above describes the conceptual pipeline.

---

## How to Run the Project
### 1) Clone the repository
```bash
git clone https://github.com/yashjajoria/mlflow-randomforest-housing.git
cd mlflow-randomforest-housing
```

### 2) Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows PowerShell
```

### 3) Install dependencies
If a `requirements.txt` exists:
```bash
pip install -r requirements.txt
```

Otherwise, install the typical dependencies:
```bash
pip install mlflow scikit-learn pandas numpy
```

### 4) Run training / experiment script
Run the main script responsible for training and logging to MLflow (replace with the correct entrypoint in your repo):
```bash
python train.py
```

If your training script is in a different path (e.g., `src/train.py`), update the command accordingly.

---

## MLflow Tracking UI Instructions
### 1) Start the MLflow UI
From the repository root:
```bash
mlflow ui
```

By default, the UI is available at:
- `http://127.0.0.1:5000`

### 2) View experiments
In the MLflow UI, you can:
- Compare runs (params/metrics)
- Inspect artifacts (model, plots, etc.)
- Review the logged **model signature**

### 3) Model Registry
If the project registers models:
- Open the **Models** tab in MLflow UI
- Inspect versions and transition stages (e.g., *Staging*, *Production*)

> If you configure a tracking server or remote artifact store later, MLflow will scale to multi-user workflows.

---

## Example Output
After a successful run, you should see:
- A new MLflow run with:
  - Logged hyperparameters from GridSearchCV best estimator
  - Metrics such as RMSE/MAE/R²
  - A saved MLflow model artifact
  - Signature information

Example (illustrative) console output:
```text
Best params: {'max_depth': 20, 'n_estimators': 300, 'min_samples_split': 2}
Test RMSE: 0.52
Test MAE: 0.36
Test R2: 0.81
Model logged to MLflow.
```

---

## Future Improvements
- Add **Docker** support for reproducible runs.
- Add a **Makefile** / task runner for common commands.
- Add **pre-commit** hooks and formatting/linting (ruff/black).
- Log additional artifacts:
  - Feature importance plots
  - Residual plots / error analysis
- Add **unit tests** for training and model loading.
- Add a **CI pipeline** (GitHub Actions) to validate training script execution.
- Deploy the registered model via:
  - `mlflow models serve`
  - a lightweight FastAPI service

---

## Author
**yashjajoria**

- GitHub: https://github.com/yashjajoria