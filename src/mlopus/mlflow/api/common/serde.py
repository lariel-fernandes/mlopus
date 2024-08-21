from pathlib import Path
from typing import Type, TypeVar

from mlopus.utils import pydantic, json_utils

T = TypeVar("T", bound=pydantic.BaseModel)


class EntitySerializer(pydantic.BaseModel):
    """Base (de)serializer for metadata entities."""

    @classmethod
    def read(cls, type_: Type[T], data: str) -> T:
        return type_.parse_obj(json_utils.loads(data))

    @classmethod
    def dump(cls, data: T) -> str:
        return json_utils.dumps(data.dict())

    def load(self, type_: Type[T], path: Path) -> T:
        return self.read(type_, data=path.read_text())

    def save(self, data: T, path: Path):
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text(self.dump(data))
