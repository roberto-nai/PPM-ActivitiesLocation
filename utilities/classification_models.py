from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, make_scorer
from sklearn.model_selection import cross_validate, StratifiedKFold
from xgboost import XGBClassifier
import pandas as pd
from typing import Union
from pathlib import Path

def ml_xgboost(
    X_train: Union[pd.DataFrame, pd.Series],
    X_test: Union[pd.DataFrame, pd.Series],
    y_train: Union[pd.Series, list],
    y_test: Union[pd.Series, list],
    outcome_column: str,
    k_fold: int = 5
) -> dict:
    """
    Trains an XGBoost classifier, performs k-fold cross-validation, and evaluates its performance.
    
    Parameters:
        X_train (Union[pd.DataFrame, pd.Series]): Training features, may include the outcome column.
        X_test (Union[pd.DataFrame, pd.Series]): Testing features, may include the outcome column.
        y_train (Union[pd.Series, list]): Labels for training data.
        y_test (Union[pd.Series, list]): Labels for testing data.
        outcome_column (str): The name of the column representing the outcome to predict.
        k_fold (int): Number of folds for cross-validation (default is 5).
    
    Returns:
        dict: A dictionary containing accuracy, F1 score, and ROC/AUC for the model, including cross-validation results for Accuracy, F1 score, and ROC/AUC.
    """
    # Remove the outcome column from the feature sets if present
    if isinstance(X_train, pd.DataFrame) and outcome_column in X_train.columns:
        X_train = X_train.drop(columns=[outcome_column])
    if isinstance(X_test, pd.DataFrame) and outcome_column in X_test.columns:
        X_test = X_test.drop(columns=[outcome_column])
    
    # Define the XGBoost classifier
    xgb_model = XGBClassifier(eval_metric='logloss', n_estimators=50)
    
    # Perform k-fold cross-validation
    skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
    
    scoring = {
        'roc_auc': 'roc_auc',
        'accuracy': make_scorer(accuracy_score),
        'f1': make_scorer(f1_score)
    }
    
    # Train the final model on the entire training set
    print("- Fitting the model...", end="")
    xgb_model.fit(X_train, y_train)
    print("done!")

    # Cross-validation (does not modify the original xgb_model)
    print("- Cross-Validation of the model...", end="")
    cv_results = cross_validate(xgb_model, X_train, y_train, cv=skf, scoring=scoring, return_train_score=False)
    print("done!")

    # Make predictions on the test set
    print("- Prediction the model...", end="")
    y_pred = xgb_model.predict(X_test)
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    print("done!")

    # Evaluate performance on the test set
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Prepare the performance metrics
    performance_metrics = {
        "Accuracy": round(accuracy, 3),
        "F1-Score": round(f1, 3),
        "ROC-AUC": round(roc_auc, 3),
        "CV Accuracy (mean)": round(cv_results['test_accuracy'].mean(), 3),
        # "Cross-Validation Accuracy (std)": round(cv_results['test_accuracy'].std(), 3),
        "CV F1-Score (mean)": round(cv_results['test_f1'].mean(), 3),
        # "Cross-Validation F1 Score (std)": round(cv_results['test_f1'].std(), 3),
        "CV ROC-AUC (mean)": round(cv_results['test_roc_auc'].mean(), 3),
        # "Cross-Validation ROC/AUC (std)": round(cv_results['test_roc_auc'].std(), 3)
    }
    
    return performance_metrics


def ml_save_metrics(metrics: dict, dataset_name: str, dir_name: str, model_name: str) -> None:
    """
    Saves the model metrics to a CSV file with the dataset and model name included.

    Parameters:
        metrics (dict): Dictionary containing the model's performance metrics.
        dataset_name (str): Name of the dataset used for training/testing the model.
        dir_name (str): Directory where the CSV file will be saved.
        model_name (str): Name of the model, included in the file name.
    
    Returns:
        None
    """
    # Add model name and dataset name to the metrics as the first two columns
    metrics_with_info = {"Model_Name": model_name, "Dataset": dataset_name, **metrics}

    # Convert the updated metrics dictionary to a DataFrame
    df_metrics = pd.DataFrame([metrics_with_info])  # Creates a single-row DataFrame

    # Construct the file path
    file_name = f"metrics_{dataset_name}_{model_name}.csv"
    file_path = Path(dir_name) / file_name

    # Save the DataFrame to a CSV file
    df_metrics.to_csv(file_path, sep=";", index=False)

    print("Metrics successfully saved to:", file_path)
