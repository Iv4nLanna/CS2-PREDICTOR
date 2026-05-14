import logging

logger = logging.getLogger(__name__)


def safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value.strip().rstrip("%"))
    except (ValueError, TypeError):
        return None


def safe_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value.strip())
    except (ValueError, TypeError):
        return None
