import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

# Set the tracking URI
mlflow.set_tracking_uri('your_tracking_uri')  # Update with your MLflow tracking URI

# Load dataset
# Assuming a pandas DataFrame 'df' is available
X = df.drop('target_column', axis=1)  # Replace 'target_column' with the actual target column name
Y = df['target_column']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define model and parameters for GridSearch
model = RandomForestRegressor(random_state=42)
parameters = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
}

# GridSearchCV setup
grid_search = GridSearchCV(estimator=model, param_grid=parameters, cv=5, scoring='neg_mean_squared_error')

# Start MLflow run
with mlflow.start_run():
    # Fit the model
    grid_search.fit(X_train, Y_train)
    best_model = grid_search.best_estimator_

    # Predictions
    predictions = best_model.predict(X_test)

    # Log model
    mlflow.sklearn.log_model(best_model, "best_randomforest_model")
    
    # Log metrics
    mse = mean_squared_error(Y_test, predictions)
    rmse = mean_squared_error(Y_test, predictions, squared=False)
    r2 = r2_score(Y_test, predictions)
    
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    # Log signature
    signature = mlflow.models.infer_signature(X_test, predictions)
    mlflow.log_model(best_model, "best_randomforest_model", signature=signature)

# End of code