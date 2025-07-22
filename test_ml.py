import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics
from ml.data import process_data

# Load sample data
df = pd.read_csv("data/census.csv")

# Prepare data for testing
cat_features = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"
]

train, _ = df.iloc[:1000], df.iloc[1000:]  # smaller subset for quick tests
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
)


def test_train_model_returns_correct_type():
    """Test if train_model returns a RandomForestClassifier."""
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)


def test_compute_model_metrics_returns_expected_types():
    """Test if compute_model_metrics returns precision, recall, fbeta as floats."""
    model = train_model(X_train, y_train)
    preds = model.predict(X_train)
    precision, recall, fbeta = compute_model_metrics(y_train, preds)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)


def test_labels_shape_match_predictions():
    """Test if predictions and labels have same length."""
    model = train_model(X_train, y_train)
    preds = model.predict(X_train)
    assert preds.shape[0] == y_train.shape[0]

