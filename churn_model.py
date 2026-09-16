"""
Churn prediction models: training, evaluation, and threshold tuning.
"""

from typing import Dict

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.segmentation import build_preprocessor


def split_data(df: pd.DataFrame, test_size: float = 0.2):
    """
    Split the dataset into train/test sets for churn prediction.

    `Cluster` is excluded here since it is not used for churn prediction.
    """
    X = df.drop(columns=["Churn", "Cluster"], errors="ignore")
    y = df["Churn"]

    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)


def evaluate_model(y_true, y_pred, y_prob) -> Dict[str, float]:
    """Compute standard classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def train_logistic_regression(X_train_processed, y_train) -> LogisticRegression:
    """Train the Logistic Regression churn model."""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_processed, y_train)
    return model


def train_random_forest(X_train_processed, y_train) -> RandomForestClassifier:
    """Train the Random Forest churn model (used for comparison)."""
    model = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1
    )
    model.fit(X_train_processed, y_train)
    return model


def tune_threshold(y_true, y_prob, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7)) -> pd.DataFrame:
    """
    Evaluate precision/recall/F1 across candidate decision thresholds,
    to support choosing a business-appropriate cutoff.
    """
    rows = []
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        rows.append(
            {
                "threshold": t,
                "precision": precision_score(y_true, y_pred_t),
                "recall": recall_score(y_true, y_pred_t),
                "f1": f1_score(y_true, y_pred_t),
            }
        )
    return pd.DataFrame(rows)


def get_feature_importance(preprocessor, model: LogisticRegression) -> pd.DataFrame:
    """
    Extract and rank feature importance from Logistic Regression coefficients.
    """
    feature_names = preprocessor.get_feature_names_out()
    coefficients = model.coef_[0]

    importance = pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
    importance["abs_coefficient"] = importance["coefficient"].abs()

    return importance.sort_values("abs_coefficient", ascending=False)


def run_churn_pipeline(df: pd.DataFrame, final_threshold: float = 0.30) -> dict:
    """
    Run the full churn prediction pipeline: split, preprocess, train
    both models, evaluate, tune threshold, and score the full dataset.

    Returns
    -------
    dict
        Contains trained models, preprocessor, metrics, feature
        importance, and the dataset with `Churn_Probability` /
        `Predicted_Churn` columns added.
    """
    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    logistic_model = train_logistic_regression(X_train_processed, y_train)
    rf_model = train_random_forest(X_train_processed, y_train)

    y_prob_log = logistic_model.predict_proba(X_test_processed)[:, 1]
    y_pred_log = logistic_model.predict(X_test_processed)
    y_prob_rf = rf_model.predict_proba(X_test_processed)[:, 1]
    y_pred_rf = rf_model.predict(X_test_processed)

    metrics_log = evaluate_model(y_test, y_pred_log, y_prob_log)
    metrics_rf = evaluate_model(y_test, y_pred_rf, y_prob_rf)

    print("Logistic Regression:", {k: round(v, 4) for k, v in metrics_log.items()})
    print("Random Forest:", {k: round(v, 4) for k, v in metrics_rf.items()})

    threshold_table = tune_threshold(y_test, y_prob_log)
    print("\nThreshold tuning:")
    print(threshold_table.round(4).to_string(index=False))

    importance = get_feature_importance(preprocessor, logistic_model)

    # Score the full dataset with the final chosen model + threshold
    X_full = df.drop(columns=["Churn", "Cluster"], errors="ignore")
    X_full_processed = preprocessor.transform(X_full)
    full_probs = logistic_model.predict_proba(X_full_processed)[:, 1]

    df = df.copy()
    df["Churn_Probability"] = full_probs
    df["Predicted_Churn"] = (full_probs >= final_threshold).astype(int)

    return {
        "logistic_model": logistic_model,
        "random_forest_model": rf_model,
        "preprocessor": preprocessor,
        "metrics_logistic": metrics_log,
        "metrics_random_forest": metrics_rf,
        "feature_importance": importance,
        "scored_dataset": df,
    }
