from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV

from cs2_predictor.pipeline.model.train import TrainedModel


@dataclass
class CalibratedModel:
    calibrator: CalibratedClassifierCV
    feature_names: list[str]
    base_accuracy: float

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.calibrator.predict_proba(X)[:, 1]


def calibrate_platt(model: TrainedModel, X: np.ndarray, y: np.ndarray) -> CalibratedModel:
    calibrator = CalibratedClassifierCV(estimator=model.estimator, method="sigmoid", cv=None)
    calibrator.fit(X, y)
    return CalibratedModel(
        calibrator=calibrator,
        feature_names=model.feature_names,
        base_accuracy=model.accuracy,
    )
