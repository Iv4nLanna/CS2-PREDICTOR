from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


@dataclass
class TrainedModel:
    estimator: LogisticRegression
    feature_names: list[str]
    accuracy: float

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(X)[:, 1]


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainedModel:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False,
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    accuracy = float(clf.score(X_val, y_val))
    return TrainedModel(estimator=clf, feature_names=feature_names, accuracy=accuracy)
