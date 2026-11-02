from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from app.policy.config import PolicyRules, load_rules

JST = ZoneInfo("Asia/Tokyo")
NEW_YORK = ZoneInfo("America/New_York")


def jst(y: int, mo: int, d: int, h: int, mi: int = 0, s: int = 0) -> datetime:
    return datetime(y, mo, d, h, mi, s, tzinfo=JST)


@pytest.fixture(scope="session")
def rules() -> PolicyRules:
    return load_rules()
