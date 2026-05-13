import numpy as np

from cs2_predictor.pipeline.model.train import (
    TrainedModel,
    train_logistic_regression,
)


def test_trained_model_predicts_probabilities():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    feature_names = ["f1", "f2", "f3", "f4"]
    model = train_logistic_regression(X, y, feature_names=feature_names)
    assert isinstance(model, TrainedModel)
    probs = model.predict_proba(X[:5])
    assert probs.shape == (5,)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_accuracy_is_reported():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 3))
    y = (X[:, 0] > 0).astype(int)
    model = train_logistic_regression(X, y, feature_names=["a", "b", "c"])
    assert model.accuracy >= 0.7
    assert model.feature_names == ["a", "b", "c"]
