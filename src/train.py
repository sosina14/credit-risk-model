# -*- coding: utf-8 -*-
"""
train.py — Credit Risk Model
Model training, hyperparameter tuning, and MLflow experiment tracking.
Author: Sosina Ayele
"""

import logging
import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TEST_SIZE = 0.2
EXPERIMENT_NAME = "credit_risk_model"

FEATURE_COLS = [
    'total_amount', 'avg_amount', 'std_amount', 'min_amount', 'max_amount',
    'total_value', 'avg_value', 'transaction_count', 'unique_products',
    'unique_channels', 'fraud_rate', 'avg_hour', 'avg_dayofweek',
    'weekend_ratio', 'Recency', 'Frequency', 'Monetary',
]


def load_data(path: str):
    """Load processed features CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed data not found: {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded: {df.shape}")
    logger.info(f"Target distribution:\n{df['is_high_risk'].value_counts()}")
    return df


def prepare_data(df: pd.DataFrame):
    """Split into train/test sets."""
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols].copy()
    y = df['is_high_risk'].copy()

    X = X.fillna(X.median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")
    logger.info(f"Train target: {y_train.value_counts().to_dict()}")
    return X_train, X_test, y_train, y_test


def evaluate(model, X_test, y_test):
    """Compute all evaluation metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        'accuracy':  round(accuracy_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'recall':    round(recall_score(y_test, y_pred, zero_division=0), 4),
        'f1':        round(f1_score(y_test, y_pred, zero_division=0), 4),
        'roc_auc':   round(roc_auc_score(y_test, y_prob), 4),
    }


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train Logistic Regression with MLflow tracking."""
    with mlflow.start_run(run_name="LogisticRegression"):
        params = {'C': 0.1, 'max_iter': 1000,
                  'random_state': RANDOM_STATE, 'class_weight': 'balanced'}

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(**params))
        ])
        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, "logistic_regression")

        logger.info(f"Logistic Regression: {metrics}")
        return pipeline, metrics


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest with RandomizedSearchCV and MLflow tracking."""
    with mlflow.start_run(run_name="RandomForest"):
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'class_weight': ['balanced'],
        }
        base_model = RandomForestClassifier(random_state=RANDOM_STATE)
        search = RandomizedSearchCV(
            base_model, param_grid, n_iter=10,
            cv=3, scoring='roc_auc',
            random_state=RANDOM_STATE, n_jobs=-1
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        metrics = evaluate(best_model, X_test, y_test)

        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, "random_forest")

        logger.info(f"Random Forest best params: {search.best_params_}")
        logger.info(f"Random Forest: {metrics}")
        return best_model, metrics


def train_gradient_boosting(X_train, y_train, X_test, y_test):
    """Train Gradient Boosting with RandomizedSearchCV and MLflow tracking."""
    with mlflow.start_run(run_name="GradientBoosting"):
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.8, 1.0],
        }
        base_model = GradientBoostingClassifier(random_state=RANDOM_STATE)
        search = RandomizedSearchCV(
            base_model, param_grid, n_iter=10,
            cv=3, scoring='roc_auc',
            random_state=RANDOM_STATE, n_jobs=-1
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        metrics = evaluate(best_model, X_test, y_test)

        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, "gradient_boosting")

        logger.info(f"Gradient Boosting best params: {search.best_params_}")
        logger.info(f"Gradient Boosting: {metrics}")
        return best_model, metrics


def run_training(data_path: str = "data/processed/features.csv"):
    """Run full training pipeline with MLflow tracking."""
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_data(data_path)
    X_train, X_test, y_train, y_test = prepare_data(df)

    results = {}

    logger.info("Training Logistic Regression...")
    lr_model, lr_metrics = train_logistic_regression(
        X_train, y_train, X_test, y_test)
    results['LogisticRegression'] = lr_metrics

    logger.info("Training Random Forest...")
    rf_model, rf_metrics = train_random_forest(
        X_train, y_train, X_test, y_test)
    results['RandomForest'] = rf_metrics

    logger.info("Training Gradient Boosting...")
    gb_model, gb_metrics = train_gradient_boosting(
        X_train, y_train, X_test, y_test)
    results['GradientBoosting'] = gb_metrics

    # Compare models
    print("\n=== Model Comparison ===")
    comparison = pd.DataFrame(results).T
    print(comparison.to_string())

    # Register best model
    best_name = comparison['roc_auc'].idxmax()
    best_metrics = comparison.loc[best_name]
    logger.info(f"\nBest model: {best_name} (AUC={best_metrics['roc_auc']:.4f})")

    models = {
        'LogisticRegression': lr_model,
        'RandomForest': rf_model,
        'GradientBoosting': gb_model,
    }
    best_model = models[best_name]

    # Save best model
    os.makedirs('models', exist_ok=True)
    import pickle
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    logger.info("Best model saved to models/best_model.pkl")

    # Register in MLflow
    with mlflow.start_run(run_name=f"BEST_{best_name}"):
        mlflow.log_params({'best_model': best_name})
        mlflow.log_metrics(best_metrics.to_dict())
        mlflow.sklearn.log_model(
            best_model, "best_model",
            registered_model_name="credit_risk_best_model"
        )

    return best_model, comparison


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/features.csv"
    best_model, comparison = run_training(path)
    print("\n=== Final Comparison ===")
    print(comparison)
