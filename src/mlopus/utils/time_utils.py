import re
from datetime import datetime
from typing import TypeVar

import pytz

T = TypeVar("T")  # Any type


class Patterns:
    """Datetime patterns."""

    SAFE_REPR = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+00:00$")  # Match output format of `safe_repr`


def to_utc(dt: datetime) -> datetime:
    """Get equivalent datetime in UTC.

    Example:
        utc_plus_2 = pytz.FixedOffset(+2 * 60)

        noon = datetime(2024, 4, 1, 12)

        noon_in_berlin = utc_plus_2.localize(noon)

        to_utc(noon_in_berlin)  # 10 AM
    """
    return dt.astimezone(pytz.utc)


def safe_repr(dt: datetime) -> str:
    """Get safe string representation of datetime (UTC ISO with microseconds precision.)

    This representation is lossless, independent of locale and preserves datetime sorting.
    """
    return to_utc(dt).isoformat(timespec="microseconds")


def maybe_parse_safe_repr(val: T) -> T | datetime:
    """Parse safe datetime representation if matches, no change otherwise."""
    if isinstance(val, str) and Patterns.SAFE_REPR.fullmatch(val):
        val = datetime.fromisoformat(val)
    return val
