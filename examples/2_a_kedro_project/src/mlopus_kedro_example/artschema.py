"""Artifact Schema for `AnnModel`."""
from pathlib import Path
from typing import Type, TypeVar, Generic

import mlopus
import numpy as np
from mlopus.utils import import_utils

from .model import AnnModel, AnnParams

Artifact = AnnModel

A = TypeVar("A", bound=Artifact)


class Dumper(mlopus.artschema.Dumper[A], Generic[A]):
    """Dumps and verifies files for `AnnModel`."""

    index_file: str = "index.bin"
    labels_file: str = "labels.npy"
    params_file: str = "params.json"

    def _dump(self, path: Path, artifact: A) -> None:
        path.mkdir()
        artifact.index.save(str(path / self.index_file))  # save index data
        np.save(path / self.labels_file, artifact.labels)  # save labels
        (path / self.params_file).write_text(artifact.params.json())  # save params

    def _verify(self, path: Path) -> None:
        assert path.is_dir()
        assert all((path / file).is_file() for file in [self.index_file, self.labels_file, self.params_file])


D = TypeVar("D", bound=Dumper)


class Loader(mlopus.artschema.Loader[A, D], Generic[A, D]):
    """Loads `AnnModel` from files."""
    model_cls: Type[A] | str = AnnModel

    def _load(self, path: Path, dumper: D) -> A:
        if isinstance(model_cls := self.model_cls, str):
            model_cls = import_utils.find_type(self.model_cls, AnnModel)

        return model_cls(
            data=(path / dumper.index_file),  # load index data
            labels=np.load(path / dumper.labels_file),  # load labels
            params=AnnParams.parse_file(path / dumper.params_file),  # load params
        )


L = TypeVar("L", bound=Loader)


class AnnModelSchema(mlopus.artschema.Schema[A, D, L], Generic[A, D, L]):
    """Artifact Schema for `AnnModel`."""
