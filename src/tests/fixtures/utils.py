import contextlib
import hashlib
import os
from pathlib import Path
from typing import Type, Iterable, Tuple, Dict, TypeVar, Iterator

import pytest

A = TypeVar("A")  # Any type


@contextlib.contextmanager
def raises_if(condition: bool, exc: Type[Exception] = Exception):
    if condition:
        with pytest.raises(exc) as exc_info:
            yield exc_info
    else:
        yield None


def random_bytes(length: int = 10) -> bytes:
    return os.urandom(length)


def random_file(file: Path, length: int = 10):
    file.parent.mkdir(exist_ok=True, parents=True)
    with open(file, "wb") as writer:
        writer.write(random_bytes(length))


def md5_file(file: Path) -> str:
    hash_md5 = hashlib.md5()

    with open(file, "rb") as reader:
        hash_md5.update(reader.read())

    return hash_md5.hexdigest()


def md5_dir(path: Path) -> Dict[str, str | dict]:
    def _md5_dir(_path: Path) -> Iterable[Tuple[str, str | dict]]:
        for dirpath, _, filenames in os.walk(_path):
            for file in filenames:
                yield str((file := Path(dirpath).joinpath(file)).relative_to(_path)), md5_file(file)

    return dict(_md5_dir(path))


def iter_one(iterator: Iterator[A]) -> A:
    if (n := len(values := list(iterator))) == 1:
        return values[0]
    raise ValueError(f"Expected exactly one item in iterator, got {n}.")
