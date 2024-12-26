import contextlib
import os
from abc import abstractmethod, ABC
from datetime import datetime
from pathlib import Path
from typing import Generic, TypeVar

import pytest

import mlopus.mlflow
from mlopus.mlflow.api.base import BaseMlflowApi
from mlopus.utils import dicts, paths
from tests.fixtures.utils import iter_one

API = TypeVar("API", bound=BaseMlflowApi)  # Type of API


class TestApi(ABC, Generic[API]):
    """Base class for testing MLflow APIs."""

    @pytest.fixture(scope="function")
    def api(self, request, temp_dir) -> API:
        with temp_dir() as tmp:
            yield self._build_api(request, tmp)

    @abstractmethod
    def _build_api(self, request, tmp: Path) -> API:
        """Build API instance."""

    @abstractmethod
    @contextlib.contextmanager
    def _ctx_exp(self, request, api: API, name: str, tags: dicts.AnyDict) -> str:
        """Context for testing with an experiment.
        Setup: Create experiment and return ID.
        Teardown: Delete experiment.
        """

    @abstractmethod
    @contextlib.contextmanager
    def _ctx_model(self, request, api: API, name: str, tags: dicts.AnyDict) -> None:
        """Context for testing with a registered model.
        Setup: Create model with name and tags.
        Teardown: Delete model.
        """

    @abstractmethod
    @contextlib.contextmanager
    def _ctx_run(self, request, api: API, exp_id: str, name: str, tags: dicts.AnyDict) -> str:
        """Context for testing with a run.
        Setup: Start run and return ID.
        Teardown: Delete run.
        """

    @abstractmethod
    @contextlib.contextmanager
    def _ctx_model_version(self, request, api: API, run_id: str, name: str, tags: dicts.AnyDict) -> str:
        """Context for testing with a model version.
        Setup: Register model version and return version number.
        Teardown: Delete model version.
        """

    def test_get_exp(self, request, api: API):
        with self._ctx_exp(request, api, "test", {}) as exp_id:
            assert api.get_exp(exp_id).id == exp_id

    def test_find_exp(self, request, api: API):
        name_query = {"name": (name := "test")}
        tags = {
            "mlflow": {"key": "val"},
            "foo": {"bar": "foobar"},
            "datetime": datetime.now(),
            "a_list": ["foo", "bar"],
            "a-b/c_d": 1,
        }

        extra_queries = [
            {"tags.a_list": "foo"},  # Relies on MongoDB matching inside documents list
        ]

        with self._ctx_exp(request, api, name, tags) as exp_id:
            tag_queries = [{"tags." + ".".join(k): v} for k, v in dicts.flatten(tags).items()]

            for i, query in enumerate([name_query] + tag_queries + extra_queries):
                print(f"Query #{i}: {query}")
                assert iter_one(api.find_exps(query)).id == exp_id

    def test_find_run(self, request, api: API):
        name_query = {"name": (name := "test")}
        status_query = {"status": mlopus.mlflow.RunStatus.RUNNING}
        tags = {
            "mlflow": {"key": "val"},
            "foo": {"bar": "foobar"},
            "date": datetime.today(),
            "datetime": datetime.now(),
            "a_list": ["foo", "bar"],
        }

        with (
            self._ctx_exp(request, api, "test", {}) as exp_id,
            self._ctx_run(request, api, exp_id, name, tags) as run_id,
        ):
            tag_queries = [{"tags." + ".".join(k): v} for k, v in dicts.flatten(tags).items()]

            for i, query in enumerate([name_query] + tag_queries + [status_query]):
                query["exp.id"] = exp_id
                print(f"Query #{i}: {query}")
                assert iter_one(api.find_runs(query)).id == run_id

    def test_find_model(self, request, api: API):
        name_query = {"name": (name := "test")}
        tags = {
            "mlflow": {"key": "val"},
            "foo": {"bar": "foobar"},
            "datetime": datetime.now(),
            "a_list": ["foo", "bar"],
            "a-b/c_d": 1,
        }

        extra_queries = [
            {"tags.a_list": "foo"},  # Relies on MongoDB matching inside documents list
        ]

        with self._ctx_model(request, api, name, tags):
            tag_queries = [{"tags." + ".".join(k): v} for k, v in dicts.flatten(tags).items()]

            for i, query in enumerate([name_query] + tag_queries + extra_queries):
                print(f"Query #{i}: {query}")
                assert iter_one(api.find_models(query)).name == name

    def test_find_mv(self, request, api: API):
        tags = {
            "mlflow": {"key": "val"},
            "foo": {"bar": "foobar"},
            "date": datetime.today(),
            "datetime": datetime.now(),
            "a_list": ["foo", "bar"],
        }

        tag_queries = [{"tags." + ".".join(k): v} for k, v in dicts.flatten(tags).items()]
        parent_run_tag_queries = [{"run.%s" % k: v for k, v in q.items()} for q in tag_queries]
        parent_model_tag_queries = [{"model.%s" % k: v for k, v in q.items()} for q in tag_queries]

        with (
            self._ctx_exp(request, api, name := "test", {}) as exp_id,
            self._ctx_run(request, api, exp_id, name, tags) as run_id,
            self._ctx_model(request, api, name, tags),
            self._ctx_model_version(request, api, run_id, name, tags) as version,
        ):
            parent_model_queries = [{"model.name": name}] + parent_model_tag_queries
            parent_run_queries = [{"run.id": run_id}, {"run.name": name}] + parent_run_tag_queries

            for i, query in enumerate(tag_queries + parent_run_queries + parent_model_queries):
                print(f"Query #{i}: {query}")
                assert iter_one(api.find_model_versions(query)).version == version

    def test_child_runs(self, request, api: API):
        with (
            self._ctx_exp(request, api, "test", {}) as exp_id,
            self._ctx_run(request, api, exp_id, "test", {}) as run_id,
        ):
            child = (parent := api.get_run(run_id)).start_child("child")
            assert iter_one(parent.children).id == child.id

    def test_artifacts(self, request, temp_dir, api: API):
        path_in_run = "run_artifact"
        artifact = {str(i): "x" * i for i in range(1, 4)}
        dumper = lambda path: path.mkdir() or [(path / k).write_text(v) for k, v in artifact.items()]  # noqa
        loader = lambda path: {f: (path / f).read_text() for f in os.listdir(path)}  # noqa

        with (
            self._ctx_exp(request, api, name := "test", {}) as exp_id,
            self._ctx_run(request, api, exp_id, name, {}) as run_id,
            self._ctx_model(request, api, name, {}),
            self._ctx_model_version(request, api, run_id, name, {}),
        ):
            api.log_run_artifact(run_id, lambda p: p.write_text("ok"), path_in_run, use_cache=True)

            with pytest.raises(PermissionError):  # Manual rewrite is not permitted
                api.get_run_artifact(run_id, path_in_run).write_text("bla")

            api.log_run_artifact(run_id, dumper, path_in_run, use_cache=True)

            assert len(api.list_run_artifacts(run_id, path_in_run)) == len(artifact)

            assert api.in_offline_mode.load_run_artifact(run_id, loader, path_in_run) == artifact

            api.clean_cached_run_artifact(run_id, path_in_run)

            with pytest.raises(FileNotFoundError):
                api.in_offline_mode.get_run_artifact(run_id, path_in_run)

            for path in (api.cache_dir, api.cache_dir / "subdir", api.cache_dir.parent):
                with pytest.raises(paths.IllegalPath):  # cannot export cache to cache, to its subdirs or parents
                    api.export_run_artifact(run_id, path)

            with temp_dir() as tmp:
                api.export_run_artifact(run_id, export := tmp / "export", path_in_run)

                exported_api = api.in_offline_mode.copy(update={"cache_dir": export})

                assert len(exported_api.list_run_artifacts(run_id, path_in_run)) == len(artifact)

                exported_api.place_run_artifact(run_id, copy := tmp / "copy", path_in_run, link=True)

                assert loader(copy) == artifact

                for dirpath, dirnames, filenames in os.walk(copy):
                    for dirname in dirnames:
                        assert not (Path(dirpath) / dirname).is_symlink()
                    for filename in filenames:
                        assert (filepath := Path(dirpath) / filename).is_symlink()
                        assert filepath.resolve().relative_to(export)

            api.clean_all_cache()
