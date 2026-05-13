from datetime import datetime, timedelta, timezone

from cs2_predictor.db.models import ModelRun


def test_accuracy_lists_versions_newest_first(client, db_session):
    now = datetime.now(timezone.utc)
    db_session.add_all([
        ModelRun(version="v1", trained_at=now - timedelta(days=2),
                 accuracy=0.62, features_used=["x"]),
        ModelRun(version="v2", trained_at=now - timedelta(days=1),
                 accuracy=0.68, features_used=["x", "y"]),
        ModelRun(version="v3", trained_at=now,
                 accuracy=0.71, features_used=["x", "y", "z"]),
    ])
    db_session.commit()
    response = client.get("/model/accuracy")
    assert response.status_code == 200
    data = response.json()
    assert [r["version"] for r in data] == ["v3", "v2", "v1"]
    assert data[0]["accuracy"] == 0.71


def test_accuracy_empty(client):
    response = client.get("/model/accuracy")
    assert response.status_code == 200
    assert response.json() == []
