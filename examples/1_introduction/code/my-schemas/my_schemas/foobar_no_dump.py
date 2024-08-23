from pathlib import Path

import mlopus
from typing import TypeVar, Generic
from . import foobar

Artifact = foobar.Artifact

A = TypeVar("A", bound=Artifact)


class Dumper(foobar.Dumper[A], Generic[A]):
    """A dumper for my model or dataset.

    It knows how to verify the dumped files, but doesn't know how to dump them.
    """

    def _dump(self, path: Path, artifact: A) -> None:
        raise NotImplementedError()


D = TypeVar("D", bound=Dumper)


class Loader(foobar.Loader[A, D], Generic[A, D]):
    """A loader for my model or dataset."""


L = TypeVar("L", bound=Loader)


class Schema(foobar.Schema[A, D, L], Generic[A, D, L]):
    """A schema for my model or dataset."""
