import numpy as np
from sklearn.linear_model import LogisticRegression

from cs2_predictor.pipeline.model.calibration import CalibratedModel, calibrate_platt
from cs2_predictor.pipeline.model.train import TrainedModel


def _trained_model():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = (X[:, 0] > 0).astype(int)
    clf = LogisticRegression(max_iter=500).fit(X, y)
    return TrainedModel(estimator=clf, feature_names=["a", "b"], accuracy=0.9), X, y


def test_calibration_returns_probabilities_in_range():
    model, X, y = _trained_model()
    calibrated = calibrate_platt(model, X, y)
    assert isinstance(calibrated, CalibratedModel)
    probs = calibrated.predict_proba(X[:10])
    assert ((probs >= 0) & (probs <= 1)).all()


def test_calibration_preserves_feature_names():
    model, X, y = _trained_model()
    calibrated = calibrate_platt(model, X, y)
    assert calibrated.feature_names == ["a", "b"]
