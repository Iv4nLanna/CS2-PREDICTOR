from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from cs2_predictor.config import get_settings


def get_engine():
    return create_engine(get_settings().database_url, pool_pre_ping=True)


def get_session_factory():
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def get_db() -> Session:
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
