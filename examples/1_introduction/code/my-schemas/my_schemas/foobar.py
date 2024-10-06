import os
from pathlib import Path
from typing import TypeVar, Generic

import mlopus
import pydantic


class Artifact(pydantic.BaseModel):
    """A container for my model or dataset."""
    some_data: dict[str, str]

    def predict(self, val: str) -> str:
        """A method for users to consume my model or dataset."""
        return self.some_data[val]


A = TypeVar("A", bound=Artifact)


class Dumper(mlopus.artschema.Dumper[A], Generic[A]):
    """A dumper for my model or dataset."""

    encoding: str = "UTF-8"

    def _dump(self, path: Path, artifact: A) -> None:
        (some_data_path := path / "some_data").mkdir(parents=True)
        for key, val in artifact.some_data.items():
            (some_data_path / f"{key}.txt").write_text(val, self.encoding)

    def _verify(self, path: Path) -> None:
        assert (some_data_path := path / "some_data").is_dir()
        assert all((some_data_path / file).is_file() for file in os.listdir(some_data_path))


D = TypeVar("D", bound=Dumper)


class Loader(mlopus.artschema.Loader[A, D], Generic[A, D]):
    """A loader for my model or dataset."""

    max_files: int | None = None

    def _load(self, path: Path, dumper: D) -> A:
        some_data = {}

        for i, file in enumerate(sorted(os.listdir(some_data_path := path / "some_data"))):
            if self.max_files is None or i < self.max_files:
                some_data[file.removesuffix(".txt")] = (some_data_path / file).read_text(dumper.encoding)

        return Artifact(some_data=some_data)


L = TypeVar("L", bound=Loader)


class Schema(mlopus.artschema.Schema[A, D, L], Generic[A, D, L]):
    """A schema for my model or dataset."""
