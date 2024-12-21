import contextlib
import os
from pathlib import Path

import pytest

from mlopus.mlflow.providers.generic import GenericMlflowApi
from mlopus.mlflow.providers.mlflow import MlflowApi
from mlopus.utils import dicts

from .mlflow_api_tester import TestApi as ApiTester
from .test_mlflow import TestMlflow as NonGenericTester

API = GenericMlflowApi


class TestGeneric(ApiTester[API]):
    """Test generic MLflow API."""

    def _build_api(self, request, tmp: Path) -> GenericMlflowApi:
        """Build generic API instance to be tested with a pre-setup cache."""
        return GenericMlflowApi(
            cache_dir=tmp.joinpath("cache"),
            cache_local_artifacts=True,
            pull_artifacts_in_offline_mode=True,
        )

    def _build_mlflow_api(self, request, api: API) -> MlflowApi:  # noqa
        """Build non-generic MLflow API instance for setting up cache fixtures."""
        return NonGenericTester()._build_api(request, tmp=api.cache_dir.parent)

    @contextlib.contextmanager
    def _ctx_exp(self, request, api: API, name: str, tags: dicts.AnyDict) -> str:
        """Context for testing with an experiment.
        Setup: Create experiment and return ID.
        Teardown: Delete experiment.
        """
        with NonGenericTester()._ctx_exp(request, api := self._build_mlflow_api(request, api), name, tags) as exp_id:
            api.get_exp(exp_id).cache_meta()
            yield exp_id

    @contextlib.contextmanager
    def _ctx_model(self, request, api: API, name: str, tags: dicts.AnyDict) -> None:
        """Context for testing with a registered model.
        Setup: Create model with name and tags.
        Teardown: Delete model.
        """
        with NonGenericTester()._ctx_model(request, api := self._build_mlflow_api(request, api), name, tags):
            api.get_model(name).cache_meta()
            yield None

    @contextlib.contextmanager
    def _ctx_run(self, request, api: API, exp_id: str, name: str, tags: dicts.AnyDict) -> str:
        """Context for testing with a run.
        Setup: Start run and return ID.
        Teardown: Delete run.
        """
        with NonGenericTester()._ctx_run(
            request, api := self._build_mlflow_api(request, api), exp_id, name, tags
        ) as run_id:
            print(api.get_run(run_id).cache_meta().status)
            for x in list(os.walk(api.cache_dir)):
                print(x)
            yield run_id

    @contextlib.contextmanager
    def _ctx_model_version(self, request, api: API, run_id: str, name: str, tags: dicts.AnyDict) -> str:
        """Context for testing with a model version.
        Setup: Register model version and return version number.
        Teardown: Delete model version.
        """
        with NonGenericTester()._ctx_model_version(
            request, api := self._build_mlflow_api(request, api), run_id, name, tags
        ) as version:
            api.get_model_version((name, version)).cache_meta()
            yield version

    @contextlib.contextmanager
    def _ctx_run_artifact(self, request, api: API, run_id: str, dumper: callable, path_in_run: str):
        (api := self._build_mlflow_api(request, api)).log_run_artifact(run_id, dumper, path_in_run, use_cache=False)
        yield None

    @pytest.mark.skip(reason="Generic API cannot create child runs.")
    def test_child_runs(self, request, api: API):
        pass

    def test_artifacts(self, request, temp_dir, api: API):
        path_in_run = "run_artifact"
        artifact = {str(i): "x" * i for i in range(1, 4)}
        api_no_pull = api.model_copy(update={"pull_artifacts_in_offline_mode": False})
        dumper = lambda path: path.mkdir() or [(path / k).write_text(v) for k, v in artifact.items()]  # noqa
        loader = lambda path: {f: (path / f).read_text() for f in os.listdir(path)}  # noqa

        with (
            self._ctx_exp(request, api, name := "test", {}) as exp_id,
            self._ctx_run(request, api, exp_id, name, {}) as run_id,
            self._ctx_model(request, api, name, {}),
            self._ctx_model_version(request, api, run_id, name, {}),
            self._ctx_run_artifact(request, api, run_id, dumper, path_in_run),
        ):
            assert len(api.list_run_artifacts(run_id, path_in_run)) == len(artifact)

            with pytest.raises(Exception) as e:
                api_no_pull.list_run_artifacts(run_id, path_in_run)
            assert str(e.value).startswith("ls operation")

            with pytest.raises(FileNotFoundError):
                api_no_pull.load_run_artifact(run_id, loader, path_in_run)

            api.cache_run_artifact(run_id, path_in_run)

            assert len(api_no_pull.list_run_artifacts(run_id, path_in_run)) == len(artifact)
            assert api_no_pull.load_run_artifact(run_id, loader, path_in_run) == artifact
