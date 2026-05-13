import pytest
from fastapi.testclient import TestClient

from cs2_predictor.api.main import create_app
from cs2_predictor.db.session import get_db


@pytest.fixture
def client(db_session):
    app = create_app()
    app.dependency_overrides[get_db] = lambda: db_session
    return TestClient(app)
