import functools
from dataclasses import is_dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from mlopus.utils import typing_utils, import_utils, pydantic, time_utils

loads = functools.partial(yaml.load, Loader=yaml.Loader)


class Dumper(yaml.Dumper):
    """Dumper that handles multiline strings, sets, types, dataclasses, datetime, Path and pydantic objects."""

    def represent_data(self, data):
        """Dumper that handles multiline strings, types, dataclasses, datetime and pydantic objects."""
        if isinstance(data, str) and "\n" in data:
            data = "\n".join([line.rstrip() for line in data.split("\n")])
            return self.represent_scalar("tag:yaml.org,2002:str", data, style="|")

        elif isinstance(data, BaseException):
            data = {"type": import_utils.fq_name(type(data)), "message": str(data)}

        elif type_ := typing_utils.as_type(data):
            data = import_utils.fq_name(type_)

        elif p_obj := pydantic.as_model_obj(data):
            data = p_obj.dict()

        elif is_dataclass(type(data)):
            data = asdict(data)

        elif isinstance(data, datetime):
            data = time_utils.safe_repr(data)

        elif isinstance(data, set):
            data = list(data)

        elif isinstance(data, Path):
            data = str(data)

        return super().represent_data(data)


def dumps(data: Any) -> str:
    """Dumper that handles multiline strings, types, dataclasses, datetime and pydantic objects."""
    return yaml.dump(data, sort_keys=False, Dumper=Dumper)
