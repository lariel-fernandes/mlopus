import contextlib
from pathlib import Path

import pytest
from mlflow import MlflowException

from mlopus.mlflow.providers.mlflow import MlflowApi
from mlopus.utils import dicts
from .mlflow_api_tester import TestApi as ApiTester
from ..test_utils.test_json import Times

API = MlflowApi


class TestMlflow(ApiTester[API]):
    """Test default MLflow API."""

    def _build_api(self, request, tmp: Path) -> MlflowApi:
        """Build API instance."""
        return MlflowApi(
            cache_id="default",
            cache_local_artifacts=True,
            cache_dir=tmp.joinpath("cache"),
            tracking_uri=tmp.joinpath("server-data"),
        )

    @contextlib.contextmanager
    def _ctx_exp(self, request, api: API, name: str, tags: dicts.AnyDict) -> str:
        """Context for testing with an experiment.
        Setup: Create experiment and return ID.
        Teardown: Delete experiment.
        """
        exp_id = api._get_client().create_experiment(name, tags=api.data_translation.preprocess_tags(tags))
        yield exp_id
        api._get_client().delete_experiment(exp_id)

    @contextlib.contextmanager
    def _ctx_model(self, request, api: API, name: str, tags: dicts.AnyDict) -> None:
        """Context for testing with a registered model.
        Setup: Create model with name and tags.
        Teardown: Delete model.
        """
        api._get_client().create_registered_model(name, tags=api.data_translation.preprocess_tags(tags))
        yield None
        api._get_client().delete_registered_model(name)

    @contextlib.contextmanager
    def _ctx_run(self, request, api: API, exp_id: str, name: str, tags: dicts.AnyDict) -> str:
        """Context for testing with a run.
        Setup: Start run and return ID.
        Teardown: Delete run.
        """
        run_id = (
            api._get_client().create_run(exp_id, None, api.data_translation.preprocess_tags(tags), name).info.run_id
        )
        yield run_id
        api._get_client().delete_run(run_id)

    @contextlib.contextmanager
    def _ctx_model_version(self, request, api: API, run_id: str, name: str, tags: dicts.AnyDict) -> str:
        """Context for testing with a model version.
        Setup: Register model version and return version number.
        Teardown: Delete model version.
        """
        tags = api.data_translation.preprocess_tags(tags)
        version = str(api._get_client().create_model_version(name, "run:/%s/%s" % (run_id, name), run_id, tags).version)
        yield version
        api._get_client().delete_model_version(name, version)

    _query_push_down_exp_scenarios = [
        (
            {"a": 1, "name": "o'k", "tags.mlflow.x": "y", "tags.foo": "bar'"},
            "name = 'o''k' AND tags.mlflow.x = 'y' AND tags.foo = '\"bar''\"'",
            {"a": 1},
            [("a", -1), ("name", 1)],
            "name ASC",
            [("a", -1)],
        ),
    ]

    @pytest.mark.parametrize("scenario", range(len(_query_push_down_exp_scenarios)))
    def test_query_push_down_exp(self, api: API, scenario: int):
        api.query_push_down.nested_subjects.add("tags")

        query, filter_expr, query_remainder, sorting, sorting_expr, sorting_remainder = (
            self._query_push_down_exp_scenarios[scenario]
        )

        actual_filter_expr, actual_query_remainder, actual_sort_expr, actual_sort_remainder = (
            api.query_push_down.parse_exp(query, sorting)
        )

        assert actual_filter_expr == filter_expr
        assert actual_query_remainder == query_remainder
        assert actual_sort_expr == sorting_expr
        assert actual_sort_remainder == sorting_remainder

    _query_push_down_run_scenarios = [
        (
            {"exp.id": "0", "tags.date": {"$gt": Times.noon_in_berlin}, "start_time": {"$lte": Times.now_here}},
            f"tags.date > '\"2024-01-01T10:00:00.000000+00:00\"' AND start_time <= {int(Times.now_here.timestamp()*1000)}",  # noqa
            {},
            [],
            "",
            [],
            ["0"],
        ),
    ]

    @pytest.mark.parametrize("scenario", range(len(_query_push_down_run_scenarios)))
    def test_query_push_down_run(self, api: API, scenario: int):
        api.query_push_down.nested_subjects.add("tags")

        query, filter_expr, query_remainder, sorting, sorting_expr, sorting_remainder, exp_ids = (
            self._query_push_down_run_scenarios[scenario]
        )

        actual_filter_expr, actual_query_remainder, actual_sort_expr, actual_sort_remainder, exp_ids = (
            api.query_push_down.parse_run(query, sorting)
        )

        assert actual_filter_expr == filter_expr
        assert actual_query_remainder == query_remainder
        assert actual_sort_expr == sorting_expr
        assert actual_sort_remainder == sorting_remainder

    _query_push_down_mv_scenarios = [
        (
            {"model.name": "test"},
            "name = 'test'",
            {},
            [],
            "",
            [],
        ),
    ]

    @pytest.mark.parametrize("scenario", range(len(_query_push_down_run_scenarios)))
    def test_query_push_down_mv(self, api: API, scenario: int):
        api.query_push_down.nested_subjects.add("tags")

        query, filter_expr, query_remainder, sorting, sorting_expr, sorting_remainder = (
            self._query_push_down_mv_scenarios[scenario]
        )

        actual_filter_expr, actual_query_remainder, actual_sort_expr, actual_sort_remainder = (
            api.query_push_down.parse_mv(query, sorting)
        )

        assert actual_filter_expr == filter_expr
        assert actual_query_remainder == query_remainder
        assert actual_sort_expr == sorting_expr
        assert actual_sort_remainder == sorting_remainder

    def test_client_settings(self, api: API):
        """Make sure independent API instances can coexist with different client settings."""
        kwargs = {"tracking_uri": "https://foo", "healthcheck": False}

        api_1 = api.copy(update={"client_settings": {"http_request_max_retries": -10}, **kwargs})
        api_2 = api.copy(update={"client_settings": {"http_request_max_retries": -20}, **kwargs})

        for i, _api in enumerate([api_1, api_2]):
            with pytest.raises(MlflowException) as exc:  # Invalid param value
                _api.get_exp("0")  # Use API to trigger exception
            assert exc.value.message.endswith(str(-10 * (i + 1)))  # -10 for api_1, -20 for api_2
