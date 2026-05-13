from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from cs2_predictor.api.schemas import ModelAccuracyEntry
from cs2_predictor.db.models import ModelRun
from cs2_predictor.db.session import get_db

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/accuracy", response_model=list[ModelAccuracyEntry])
def accuracy(db: Session = Depends(get_db)):
    rows = db.query(ModelRun).order_by(ModelRun.trained_at.desc()).all()
    return [
        ModelAccuracyEntry(
            version=r.version,
            trained_at=r.trained_at,
            accuracy=r.accuracy,
            features_used=r.features_used,
        )
        for r in rows
    ]
