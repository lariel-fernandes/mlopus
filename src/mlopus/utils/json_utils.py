import json
from dataclasses import is_dataclass, asdict
from datetime import datetime, date
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict

import pydantic

from mlopus.utils import time_utils


def fallback(obj: Any) -> Dict[str, Any] | list | str | None:
    """json.dumps fallback for types, dataclasses, pydantic, date and datetime."""
    match obj:
        case Path():
            return str(obj)
        case set():
            return sorted(obj)
        case type():
            return f"{obj.__module__}.{obj.__class__}"
        case pydantic.BaseModel():
            return obj.dict()
        case _ if is_dataclass(obj):
            return asdict(obj)
        case datetime():
            # Get UTC timestamp in full precision ISO format with timezone info: 2024-04-01T09:00:00.000000+00:00
            # This preserves datetime sorting when applying alphanumerical sorting to string representation
            return time_utils.safe_repr(obj)
        case date():
            return obj.isoformat()
        case Enum():
            return obj.value


def dumps(obj: Any, fallback_: Callable[[Any], Dict[str, Any] | str | None] = fallback, **kwargs) -> str:
    """JSON encode with `fallback` function to handle encoding of special types."""
    return json.dumps(obj, default=fallback_, **kwargs)


class Decoder(json.JSONDecoder):
    """Custom JSON decoder. Handles datetime strings in safe representation (see `time_utils.safe_repr`)."""

    def decode(self, s, *args, **kwargs):
        """Decode JSON value."""
        return time_utils.maybe_parse_safe_repr(super().decode(s, *args, **kwargs))


def loads(data: str, ignore_errors: bool = False, cls=Decoder) -> Any:
    """JSON decode."""
    try:
        return json.loads(data, cls=cls)
    except json.JSONDecodeError as error:
        if ignore_errors:
            return str(data)
        raise error
