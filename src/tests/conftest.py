import contextlib
import functools
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from mlopus.mlflow.providers.mlflow import MlflowApi


@pytest.fixture(scope="session")
def temp_dir():
    @contextlib.contextmanager
    def _temp_dir(**kwargs) -> Path:
        with TemporaryDirectory(**kwargs) as tmp:
            yield Path(tmp)

    with TemporaryDirectory(prefix="pytest_") as pytest_tmp:
        yield functools.partial(_temp_dir, dir=pytest_tmp)


@pytest.fixture(scope="function")
def temp_mlflow(temp_dir):
    with temp_dir() as tmp:
        yield MlflowApi(
            cache_id="default",
            cache_local_artifacts=True,
            cache_dir=tmp.joinpath("cache"),
            tracking_uri=tmp.joinpath("server-data"),
        )
