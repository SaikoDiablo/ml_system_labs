from pathlib import Path

import joblib
import numpy as np
import pytest

from sklearn.datasets import load_breast_cancer


# Find the saved models in the results folder.
ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT_DIR / "lab01" / "results"

MODEL_PATHS = [
    RESULTS_DIR / "logistic_regression.joblib",
    RESULTS_DIR / "random_forest.joblib",
]


@pytest.mark.parametrize("model_path", MODEL_PATHS)
def test_saved_model_exists(model_path):
    """Check that the trained model file exists."""
    assert model_path.exists()


@pytest.mark.parametrize("model_path", MODEL_PATHS)
def test_model_can_make_predictions(model_path):
    """Check that the saved model loads and predicts correctly."""

    model = joblib.load(model_path)

    dataset = load_breast_cancer(as_frame=True)
    sample = dataset.data.iloc[[0]]

    prediction = model.predict(sample)

    assert len(prediction) == 1
    assert prediction[0] in [0, 1]


@pytest.mark.parametrize("model_path", MODEL_PATHS)
def test_model_prediction_is_repeatable(model_path):
    """Check that repeated predictions for the same sample agree."""

    model = joblib.load(model_path)

    dataset = load_breast_cancer(as_frame=True)
    sample = dataset.data.iloc[[0]]

    first_prediction = model.predict(sample)
    second_prediction = model.predict(sample)

    assert np.array_equal(first_prediction, second_prediction)