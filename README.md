# Machine-Learning-Algorithms
## Table of Contents
- [Preprocessing Steps](#preprocessing-steps)
- [Linear Regression](#linear-regression)
- [Cross-Validation](#cross-validation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Model Training](#model-training)
- [Model Metrics](#model-metrics)

## Preprocessing Steps

Preprocessing is crucial for building robust machine learning models. Common steps include:

1. **Data Collection:** Gather raw data from sources such as databases, CSV files, or APIs.
2. **Data Cleaning:**  
    - Handle missing values (imputation, removal).
    - Remove duplicate records.
    - Correct inconsistencies and errors.
    - **Detect and handle outliers** (e.g., using z-score, IQR, or visualization techniques).
3. **Exploratory Data Analysis (EDA):**  
    - Visualize distributions and relationships.
    - Summarize statistics.
    - Identify patterns and anomalies.
4. **Feature Engineering:**  
    - Create new features from existing data.
    - Transform variables (e.g., log, polynomial).
5. **Feature Selection:**  
    - Select relevant features using statistical tests or model-based methods.
    - Remove irrelevant or redundant features.
6. **Feature Scaling:**  
    - Normalize (Min-Max scaling) or standardize (Z-score) numerical features.
    - Ensure features are on similar scales for model convergence.
7. **Encoding Categorical Variables:**  
    - Apply one-hot encoding for nominal categories.
    - Use label encoding for ordinal categories.
8. **Handling Imbalanced Data:**  
    - Apply resampling techniques (oversampling, undersampling).
    - Use synthetic data generation (SMOTE).
9. **Train-Test Split:**  
    - Divide data into training and testing sets (commonly 70/30 or 80/20 split).
    - Optionally, create a validation set for hyperparameter tuning.

## Linear Regression

Linear regression models the relationship between a dependent variable and one or more independent variables using a linear equation.

**Equation:**  
`y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε`

Where:
- `y`: Target variable
- `x₁...xₙ`: Features
- `β₀...βₙ`: Coefficients (weights)
- `ε`: Error term

**Types:**
- **Simple Linear Regression:** One independent variable.
- **Multiple Linear Regression:** Multiple independent variables.

## Cross-Validation

Cross-validation helps assess model generalization and avoid overfitting.

**Common methods:**
- **K-Fold Cross-Validation:**  
  - Split data into k subsets (folds).
  - Train on k-1 folds, test on the remaining fold.
  - Repeat k times, average results.
- **Leave-One-Out Cross-Validation (LOOCV):**  
  - Each sample is used once as a test set, rest as training.
- **Stratified K-Fold:**  
  - Ensures each fold has a representative distribution of target classes.

## Hyperparameter Tuning

Optimize model parameters to improve performance.

**Techniques:**
- **Grid Search:**  
  - Exhaustively test all combinations of specified hyperparameters.
- **Random Search:**  
  - Randomly sample hyperparameter combinations.
- **Automated Search:**  
  - Use tools like `scikit-learn`'s `GridSearchCV` or `RandomizedSearchCV`.
- **Bayesian Optimization:**  
  - Model-based search for optimal hyperparameters.

## Model Training

Detailed steps for training a linear regression model:

1. **Preprocess Data:**  
    - Apply all relevant preprocessing steps.
2. **Split Data:**  
    - Divide into training and test sets.
3. **Instantiate Model:**  
    - Create a linear regression model object (e.g., `LinearRegression()` in scikit-learn).
4. **Fit Model:**  
    - Train the model on the training data.
5. **Cross-Validate:**  
    - Evaluate model using cross-validation techniques.
6. **Hyperparameter Tuning:**  
    - Adjust model parameters for optimal performance.
7. **Evaluate Model:**  
    - Assess performance on test data using metrics.

## Model Metrics

Evaluate model performance using:

- **Mean Squared Error (MSE):**  
  - Average squared difference between predicted and actual values.
- **Root Mean Squared Error (RMSE):**  
  - Square root of MSE, interpretable in original units.
- **Mean Absolute Error (MAE):**  
  - Average absolute difference between predicted and actual values.
- **R² Score (Coefficient of Determination):**  
  - Proportion of variance explained by the model (ranges from 0 to 1).
- **Adjusted R²:**  
  - Adjusts R² for the number of predictors, penalizing unnecessary features.


