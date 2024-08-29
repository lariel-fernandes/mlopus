import contextlib
from typing import TypeVar

import pytest
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

import mlopus
from mlopus.utils import packaging

A = TypeVar("A")  # Any type


@contextlib.contextmanager
def using(val: A) -> A:
    yield val


@pytest.fixture(scope="session")
def example_proj() -> str:
    return "examples/2_a_kedro_project"


@pytest.fixture(scope="function")
def temp_mlflow_overrides(mocker, temp_dir, example_proj) -> dict:
    bootstrap_project(example_proj)
    mocker.patch.object(packaging, "get_dist", return_value=packaging.get_dist("mlopus"))

    with (
        temp_dir() as tmp,
        using(tmp.joinpath("cache")) as cache,
        using(tmp.joinpath("server-data")) as server,
        using({"cache_dir": cache, "tracking_uri": server}) as api_conf,
        using({"globals": {"mlflow": {"api": {"conf": api_conf}}}}) as overrides,
    ):
        yield overrides


@pytest.fixture(scope="function")
def temp_session(example_proj, temp_mlflow_overrides) -> KedroSession:
    with KedroSession.create(example_proj, extra_params=temp_mlflow_overrides) as session:
        yield session


@pytest.fixture(scope="function")
def temp_kedro_conf(temp_session):
    yield temp_session.load_context().config_loader


@pytest.fixture(scope="function")
def temp_run_manager(temp_kedro_conf) -> mlopus.mlflow.traits.MlflowRunManager:
    yield mlopus.mlflow.traits.MlflowRunManager(**temp_kedro_conf["globals"]["mlflow"])
