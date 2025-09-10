import os
"""
This script performs linear regression modeling on the California Housing dataset using scikit-learn and MLflow for experiment tracking.

Workflow:
1. Loads the California Housing dataset.
2. Logs a sample of the raw data to MLflow.
3. Handles missing values and logs the process.
4. Encodes categorical features (if any) and logs the process.
5. Performs feature scaling using StandardScaler.
6. Logs a sample of the processed data to MLflow.
7. Splits the data into training and test sets.
8. Sets up an MLflow experiment for tracking.
9. Trains a Linear Regression model and logs cross-validation metrics.
10. Evaluates the model on the test set and logs performance metrics (MSE, MAE, R2).
11. Generates and logs visualizations: actual vs predicted prices and residuals distribution.
12. Performs hyperparameter tuning using Ridge regression and logs the best parameters and score.
13. Saves and logs the trained model and scaler as artifacts.
14. Prints completion message.

Dependencies:
- numpy
- pandas
- seaborn
- matplotlib
- scikit-learn
- mlflow

Artifacts and metrics are logged to MLflow for experiment tracking and reproducibility.
"""
import pickle
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import mlflow.sklearn
import matplotlib.pyplot as plt

# Load dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# Log raw data sample
mlflow.log_text(df.head().to_csv(index=False), "raw_data_sample.csv")

# Data preprocessing
# 1. Handle missing values
missing_count = df.isnull().sum().sum()
mlflow.log_param("missing_values_count", missing_count)
if missing_count > 0:
    df = df.fillna(df.median())
    mlflow.log_param("missing_values_handled", True)
else:
    mlflow.log_param("missing_values_handled", False)

# 2. Encode categorical features (none in this dataset, but for generalization)
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
mlflow.log_param("categorical_columns", list(categorical_cols))
if len(categorical_cols) > 0:
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
    df = pd.concat([df.drop(categorical_cols, axis=1), encoded_df], axis=1)
    mlflow.log_param("categorical_encoding", "onehot")
else:
    mlflow.log_param("categorical_encoding", "none")

# 3. Feature selection (optional, not applied here)
mlflow.log_param("feature_selection", "none")

X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
mlflow.log_param("scaling_method", "StandardScaler")

# Log processed data sample
processed_df = pd.DataFrame(X_scaled, columns=X.columns)
processed_df['MedHouseVal'] = y
mlflow.log_text(processed_df.head().to_csv(index=False), "processed_data_sample.csv")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
mlflow.log_param("train_test_split", "0.8/0.2")

# MLflow experiment setup
mlflow.set_experiment("LinearRegression_HousePrice")

mlflow.end_run()  # End any previous run before starting a new one

with mlflow.start_run():
    # Log train-test split sizes
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))

    # Linear Regression model
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Cross-validation
    cv_scores = cross_val_score(lr, X_train, y_train, cv=5, scoring='r2')
    mlflow.log_metric("cv_mean_r2", np.mean(cv_scores))
    mlflow.log_metric("cv_std_r2", np.std(cv_scores))

    # Predictions
    y_pred = lr.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mlflow.log_metric("test_mse", mse)
    mlflow.log_metric("test_mae", mae)
    mlflow.log_metric("test_r2", r2)

    # Visualizations
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted Prices")
    plot_path = "actual_vs_pred.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.histplot(y_test - y_pred, bins=50, kde=True)
    plt.xlabel("Residuals")
    plt.title("Residuals Distribution")
    res_plot_path = "residuals.png"
    plt.savefig(res_plot_path)
    mlflow.log_artifact(res_plot_path)
    plt.close()

    # Hyperparameter tuning with Ridge regression
    ridge = Ridge()
    param_grid = {"alpha": [0.01, 0.1, 1, 10, 100]}
    grid = GridSearchCV(ridge, param_grid, cv=5, scoring='r2')
    grid.fit(X_train, y_train)
    mlflow.log_param("best_ridge_alpha", grid.best_params_['alpha'])
    mlflow.log_metric("best_ridge_cv_r2", grid.best_score_)

    # Save model as pickle
    model_path = "linear_regression_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(lr, f)
    mlflow.log_artifact(model_path)

    # Save scaler
    scaler_path = "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    mlflow.log_artifact(scaler_path)

print("Run complete. Check MLflow UI for details.")