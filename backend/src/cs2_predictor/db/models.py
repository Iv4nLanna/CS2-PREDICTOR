import enum
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class MatchFormat(str, enum.Enum):
    BO1 = "BO1"
    BO3 = "BO3"
    BO5 = "BO5"


class MatchStatus(str, enum.Enum):
    SCHEDULED = "scheduled"
    LIVE = "live"
    FINISHED = "finished"
    CANCELLED = "cancelled"


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(primary_key=True)
    hltv_id: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    country: Mapped[str | None] = mapped_column(String(100), nullable=True)
    hltv_ranking: Mapped[int | None] = mapped_column(Integer, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class Player(Base):
    __tablename__ = "players"

    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    hltv_id: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    role: Mapped[str | None] = mapped_column(String(50), nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    faceit_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    faceit_rating: Mapped[float | None] = mapped_column(Float, nullable=True)


class Match(Base):
    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(primary_key=True)
    hltv_id: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    team_a_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    team_b_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    format: Mapped[MatchFormat] = mapped_column(Enum(MatchFormat))
    is_lan: Mapped[bool] = mapped_column(Boolean, default=False)
    map_pool: Mapped[list] = mapped_column(JSON, default=list)
    tournament: Mapped[str | None] = mapped_column(String(200), nullable=True)
    scheduled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[MatchStatus] = mapped_column(Enum(MatchStatus), default=MatchStatus.SCHEDULED)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    team_a = relationship("Team", foreign_keys=[team_a_id])
    team_b = relationship("Team", foreign_keys=[team_b_id])


class MatchResult(Base):
    __tablename__ = "match_results"
    __table_args__ = (UniqueConstraint("match_id", name="uq_match_result_match"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"))
    winner_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    score_detail: Mapped[dict] = mapped_column(JSON, default=dict)
    played_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class TeamFeatures(Base):
    __tablename__ = "team_features"
    __table_args__ = (UniqueConstraint("team_id", "match_id", name="uq_team_features_team_match"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"))
    win_rate_recent_decayed: Mapped[float] = mapped_column(Float)
    head_to_head_decayed: Mapped[float] = mapped_column(Float)
    hltv_ranking_snapshot: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sos_score: Mapped[float] = mapped_column(Float)
    map_stats: Mapped[dict] = mapped_column(JSON, default=dict)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class ModelRun(Base):
    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    version: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    accuracy: Mapped[float] = mapped_column(Float)
    features_used: Mapped[list] = mapped_column(JSON, default=list)


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"))
    team_a_win_prob: Mapped[float] = mapped_column(Float)
    team_b_win_prob: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(50))
    calibrated: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    match = relationship("Match")
