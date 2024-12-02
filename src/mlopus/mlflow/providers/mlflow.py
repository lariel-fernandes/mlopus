import contextlib
import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, TypeVar, Callable, Tuple, List, Literal, Mapping, Iterable, Set, Iterator, ContextManager

from mlflow import MlflowClient as _NativeMlflowClient, entities as native
from mlflow.entities import model_registry as native_model_registry
from mlflow.store.entities import PagedList
from mlflow.tracking import artifact_utils as mlflow_artifact_utils
from mlflow.utils import rest_utils as mlflow_rest_utils

from mlopus.mlflow.api.base import BaseMlflowApi
from mlopus.mlflow.api.common import schema, patterns
from mlopus.utils import pydantic, mongo, dicts, json_utils, iter_utils, urls, string_utils, env_utils

logger = logging.getLogger(__name__)

A = TypeVar("A")  # Any type
A2 = TypeVar("A2")  # Any type

# Entity types used in MLOpus
E = schema.Experiment
R = schema.Run
M = schema.Model
V = schema.ModelVersion
T = TypeVar("T", bound=schema.BaseEntity)

# Types used natively by open source MLflow
NE = native.Experiment
NR = native.Run
NM = native_model_registry.RegisteredModel
NV = native_model_registry.ModelVersion
NT = TypeVar("NT", NE, NR, NM, NV)


class MlflowClient(_NativeMlflowClient):
    """Patch of native MlflowClient."""

    def healthcheck(self):
        """Check if service is available."""
        if (url := urls.parse_url(self.tracking_uri)).scheme in ("http", "https"):
            creds = self._tracking_client.store.get_host_creds()
            response = mlflow_rest_utils.http_request(creds, "/health", "GET")
            assert response.status_code == 200, f"Healthcheck failed for URI '{url}'"
        elif not urls.is_local(url):
            raise NotImplementedError(f"No healthcheck for URI '{url}'")

    def get_artifact_uri(self, run_id: str) -> str:
        """Get root URL of remote run artifacts, according to MLflow server."""
        return mlflow_artifact_utils.get_artifact_uri(run_id, tracking_uri=self.tracking_uri)

    @property
    def tracking_client(self):
        """Public access to tracking client."""
        return self._tracking_client


class KeepUntouched(pydantic.BaseModel):
    """Set of rules for keys of params/values/metrics whose values are to be kept untouched.

    Their values are exchanged to/from MLflow server without any pre-processing/encoding/escaping.
    """

    prefixes: Dict[str, Set[str]] = {"tags": {"mlflow"}}

    def __call__(self, key_parts: Tuple[str, ...], scope: Literal["tags", "params", "metrics"]) -> bool:
        return key_parts[0] in self.prefixes.get(scope, ())


class MaxLength(pydantic.BaseModel):
    """Max value lengths."""

    tag: int = 500
    param: int = 500


class MlflowDataTranslation(pydantic.BaseModel):
    """Translates native MLflow data to MLOpus format and back."""

    ignore_json_errors: bool = True
    max_length: MaxLength = MaxLength()
    keep_untouched: KeepUntouched = KeepUntouched()

    class Config(BaseMlflowApi.Config):
        """Class constants."""

        STATUS_TO_MLFLOW: Dict[schema.RunStatus, native.RunStatus] = {
            schema.RunStatus.FAILED: native.RunStatus.FAILED,
            schema.RunStatus.RUNNING: native.RunStatus.RUNNING,
            schema.RunStatus.FINISHED: native.RunStatus.FINISHED,
            schema.RunStatus.SCHEDULED: native.RunStatus.SCHEDULED,
        }

        STATUS_FROM_MLFLOW: Dict[native.RunStatus, schema.RunStatus | None] = {
            **{v: k for k, v in STATUS_TO_MLFLOW.items()},
            native.RunStatus.KILLED: schema.RunStatus.FAILED,
        }

    # =======================================================================================================
    # === Datetime translation ==============================================================================

    @classmethod
    def mlflow_ts_to_datetime(cls, timestamp: int | None) -> datetime | None:
        """Parse MLflow timestamp as datetime."""
        return None if timestamp is None else datetime.fromtimestamp(timestamp / 1000)

    @classmethod
    def datetime_to_mlflow_ts(cls, datetime_: datetime) -> int:
        """Coerce datetime to MLflow timestamp."""
        return int(datetime_.timestamp() * 1000)

    # =======================================================================================================
    # === Enum translation ==================================================================================

    @classmethod
    def run_status_from_mlflow(cls, status: native.RunStatus | str) -> schema.RunStatus:
        """Parse run status enum value from MLflow."""
        status = native.RunStatus.from_string(status) if isinstance(status, str) else status
        return cls.Config.STATUS_FROM_MLFLOW[status]

    @classmethod
    def run_status_to_mlflow(cls, status: schema.RunStatus, as_str: bool = False) -> native.RunStatus | str:
        """Coerce run status enum value to MLflow format."""
        status = cls.Config.STATUS_TO_MLFLOW[status]
        return native.RunStatus.to_string(status) if as_str else status

    # =======================================================================================================
    # === Dict translation ==================================================================================

    @classmethod
    @string_utils.retval_matches(patterns.TAG_PARAM_OR_METRIC_KEY)
    def encode_key(cls, key: str) -> str:
        """Make sure dict key is safe for storage and query."""
        return str(key)

    @classmethod
    def _decode_key(cls, key: str) -> str:
        """Inverse func of _encode_key."""
        return key

    @classmethod
    def flatten_key(cls, key_parts: Iterable[str]) -> str:
        """Turns a nested dict key into a flat one by joining with dot delimiter."""
        return ".".join(key_parts)

    @classmethod
    def unflatten_key(cls, key: str) -> Tuple[str, ...]:
        """Inverse func of _flatten_key."""
        return tuple(key.split("."))

    def preprocess_key(self, key_parts: Tuple[str, ...]) -> str:
        """Process key of tag, param or metric to be sent to MLflow."""
        return self.flatten_key(self.encode_key(k) for k in key_parts)

    def _preprocess_dict(
        self, data: Mapping[str, A], val_mapper: Callable[[Tuple[str, ...], A], A2 | None]
    ) -> Iterable[Tuple[str, A2]]:
        for key_parts, val in dicts.flatten(data).items():
            if (mapped_val := val_mapper(key_parts, val)) is not None:
                yield self.preprocess_key(key_parts), mapped_val

    def preprocess_dict(
        self, data: Mapping[str, A], val_mapper: Callable[[Tuple[str, ...], A], A2 | None]
    ) -> Dict[str, A2]:
        """Pre-process dict of tags, params or metrics to be compatible with MLflow.

        - Flatten nested keys as tuples
        - URL-encode dots and other forbidden chars in keys
        - Join tuple keys as dot-delimited strings
        - Map leaf-values (tags and params as json, metrics as float)
        """
        return dict(self._preprocess_dict(data, val_mapper))

    def _deprocess_dict(
        self, data: Mapping[str, A], val_mapper: Callable[[Tuple[str, ...], A], A2]
    ) -> Iterable[Tuple[Tuple[str, ...], A2]]:
        """Inverse func of _preprocess_dict."""
        for key, val in data.items():
            key_parts = tuple(self._decode_key(k) for k in self.unflatten_key(key))
            yield key_parts, val_mapper(key_parts, val)

    def deprocess_dict(self, data: Mapping[str, A], val_mapper: Callable[[Tuple[str, ...], A], A2]) -> Dict[str, A2]:
        """Inverse func of _preprocess_dict."""
        return dicts.unflatten(self._deprocess_dict(data, val_mapper))

    def process_tag(self, key_parts: Tuple[str, ...], val: Any) -> str | None:
        """JSON encode user-tag."""
        if not self.keep_untouched(key_parts, scope="tags"):
            val = string_utils.escape_sql_single_quote(json_utils.dumps(val))
        if len(str(val)) > self.max_length.tag:
            val = None
            logger.warning("Ignoring tag above max length of %s: %s", self.max_length.tag, key_parts)
        return val

    def _deprocess_tag(self, key_parts: Tuple[str, ...], val: str) -> Any:
        """JSON decode user-tag value from MLflow."""
        if not self.keep_untouched(key_parts, scope="tags"):
            val = json_utils.loads(string_utils.unscape_sql_single_quote(val), ignore_errors=self.ignore_json_errors)
        return val

    def process_param(self, key_parts: Tuple[str, ...], val: Any) -> str:  # noqa
        """JSON encode param."""
        if not self.keep_untouched(key_parts, scope="params"):
            val = string_utils.escape_sql_single_quote(json_utils.dumps(val))
        if len(str(val)) > self.max_length.param:
            val = None
            logger.warning("Ignoring param above max length of %s: %s", self.max_length.param, key_parts)
        return val

    def _deprocess_param(self, key_parts: Tuple[str, ...], val: str) -> Any:  # noqa
        """JSON decode param value from MLflow."""
        if not self.keep_untouched(key_parts, scope="params"):
            val = json_utils.loads(string_utils.unscape_sql_single_quote(val), ignore_errors=self.ignore_json_errors)
        return val

    def process_metric(self, key_parts: Tuple[str, ...], val: Any) -> float:  # noqa
        """Coerce metric value to float."""
        if not self.keep_untouched(key_parts, scope="metrics"):
            val = float(val)
        return val

    def _deprocess_metric(self, key_parts: Tuple[str, ...], val: Any) -> float:  # noqa
        """Coerce metric val from MLflow to float."""
        if not self.keep_untouched(key_parts, scope="metrics"):
            val = float(val)
        return val

    def preprocess_tags(self, data: Mapping) -> Dict[str, str]:
        """Prepare tags dict to be sent to native MLflow API."""
        return self.preprocess_dict(data, val_mapper=self.process_tag)

    def deprocess_tags(self, data: Mapping) -> Dict[str, Any]:
        """Inverse func of _preprocess_tags."""
        return self.deprocess_dict(data, val_mapper=self._deprocess_tag)

    def preprocess_params(self, data: Mapping) -> Dict[str, str]:
        """Prepare params dict to be sent to native MLflow API."""
        return self.preprocess_dict(data, val_mapper=self.process_param)

    def deprocess_params(self, data: Mapping) -> Dict[str, Any]:
        """Inverse func of _preprocess_params."""
        return self.deprocess_dict(data, val_mapper=self._deprocess_param)

    def preprocess_metrics(self, data: Mapping) -> Dict[str, float]:
        """Prepare metrics dict to be sent to native MLflow API."""
        return self.preprocess_dict(data, val_mapper=self.process_metric)

    def deprocess_metrics(self, data: Mapping) -> Dict[str, Any]:
        """Inverse func of _preprocess_metrics."""
        return self.deprocess_dict(data, val_mapper=self._deprocess_metric)


class MlflowQueryPushDown(mongo.Mongo2Sql):
    """Basic MongoDB query to MLflow SQL conversor."""

    mongo_subj_2_mlflow: Dict[str, Dict[str, str]] = {
        "exp": {
            "name": "name",
        },
        "run": {
            "id": "id",
            "name": "run_name",
            "status": "status",
            "end_time": "end_time",
            "start_time": "start_time",
        },
        "model": {
            "name": "name",
        },
        "mv": {
            "run.id": "run_id",
            "model.name": "name",
            "version": "version_number",
        },
    }
    data_translation: MlflowDataTranslation = None
    nested_subjects: Set[str] = {"metrics"}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if intersect := self.nested_subjects.intersection({"tags", "params"}):
            logger.warning(
                "Pushing down queries for the nested subject(s) %s may produce "
                "incomplete results because of MLflow SQL limitations over JSON fields.",
                intersect,
            )

    def _parse_subj(self, coll: str, subj: str) -> str | None:
        if (scope := (parts := subj.split("."))[0]) in self.nested_subjects:
            subj = f"{scope}.%s" % self.data_translation.preprocess_key(tuple(parts[1:]))
        else:
            subj = self.mongo_subj_2_mlflow[coll].get(subj)
        return super()._parse_subj(coll, subj)

    def _parse_obj(self, coll: str, subj: str, pred: Any, raw_obj: Any) -> str | None:
        if coll == "run" and subj in ("start_time", "end_time") and isinstance(raw_obj, datetime):
            raw_obj = self.data_translation.datetime_to_mlflow_ts(raw_obj)
        elif isinstance(raw_obj, schema.RunStatus):
            raw_obj = self.data_translation.run_status_to_mlflow(raw_obj, as_str=True)
        elif process := {
            "tags": self.data_translation.process_tag,
            "params": self.data_translation.process_param,
            "metrics": self.data_translation.process_metric,
        }.get((parts := subj.split("."))[0]):
            return "'%s'" % process(tuple(parts[1:]), raw_obj)  # noqa

        return super()._parse_obj(coll, subj, pred, raw_obj)

    def parse_exp(self, query: mongo.Query, sorting: mongo.Sorting) -> Tuple[str, mongo.Query, str, mongo.Sorting]:
        """Parse query and sorting rule for experiments search, return SQL expression and remainder for each."""
        return *self.parse_query(query, coll="exp"), *self.parse_sorting(sorting, coll="exp")  # noqa

    def parse_run(
        self, query: mongo.Query, sorting: mongo.Sorting
    ) -> Tuple[str, mongo.Query, str, mongo.Sorting, List[str]]:
        """Parse query and sorting rule for runs search, return SQL expression and remainder for each, plus exp IDs."""
        if isinstance(exp_ids := query.pop("exp.id", None), str):
            exp_ids = [exp_ids]
        elif isinstance(exp_ids, dict) and set(exp_ids.keys()) == "$in":
            exp_ids = exp_ids.pop("$in")
        if not (isinstance(exp_ids, (list, tuple)) and len(exp_ids) > 0 and all(isinstance(x, str) for x in exp_ids)):
            raise ValueError(
                f"{self.__class__.__name__}: `exp.id` must be specified when querying runs. "
                'Example: {"exp.id": {"$in": ["42", "13"]}}'
            )
        return *self.parse_query(query, coll="run"), *self.parse_sorting(sorting, coll="run"), exp_ids  # noqa

    def parse_model(self, query: mongo.Query, sorting: mongo.Sorting) -> Tuple[str, mongo.Query, str, mongo.Sorting]:
        """Parse query and sorting rule for models search, return SQL expression and remainder for each."""
        return *self.parse_query(query, coll="model"), *self.parse_sorting(sorting, coll="model")  # noqa

    def parse_mv(self, query: mongo.Query, sorting: mongo.Sorting) -> Tuple[str, mongo.Query, str, mongo.Sorting]:
        """Parse query and sorting rule for model version search, return SQL expression and remainder for each."""
        return *self.parse_query(query, coll="mv"), *self.parse_sorting(sorting, coll="mv")  # noqa


class MlflowTagKeys(pydantic.BaseModel):
    """Default tag keys."""

    artifacts_repo: str = "artifacts_repo"
    parent_run_id: str = "mlflow.parentRunId"


class MlflowApi(BaseMlflowApi):
    """MLflow API provider based on open source MLflow.

    **Plugin name:** `mlflow`

    **Requires extras:** `mlflow`

    **Default cache dir:** `~/.cache/mlopus/mlflow-providers/mlflow/<hashed-tracking-uri>`

    Assumptions:
     - No artifacts proxy.
     - SQL database is server-managed.
    """

    tracking_uri: str = pydantic.Field(
        default=None,
        description=(
            "MLflow server URL or path to a local directory. "
            "Defaults to the environment variable `MLFLOW_TRACKING_URI`, "
            "falls back to `~/.cache/mlflow`."
        ),
    )

    healthcheck: bool = pydantic.Field(
        default=True,
        description=(
            "If true and not in :attr:`~mlopus.mlflow.BaseMlflowApi.offline_mode`, "
            "eagerly attempt connection to the server after initialization."
        ),
    )

    client_settings: Dict[str, str | int] = pydantic.Field(
        default_factory=dict,
        description=(
            "MLflow client settings. Keys are like the open-source MLflow environment variables, "
            "but lower case and without the `MLFLOW_` prefix. Example: `http_request_max_retries`. "
            "See: https://mlflow.org/docs/latest/python_api/mlflow.environment_variables.html"
        ),
    )

    tag_keys: MlflowTagKeys = pydantic.Field(
        repr=False,
        default_factory=MlflowTagKeys,
        description="Tag keys for storing internal information such as parent run ID.",
    )

    query_push_down: MlflowQueryPushDown = pydantic.Field(
        repr=False,
        default_factory=MlflowQueryPushDown,
        description=(
            "Utility for partial translation of MongoDB queries to open-source MLflow SQL. "
            "Users may replace this with a different implementation when subclassing the API."
        ),
    )

    data_translation: MlflowDataTranslation = pydantic.Field(
        repr=False,
        default_factory=MlflowDataTranslation,
        description=(
            "Utility for translating keys and values from MLOpus schema to native MLflow schema and back. "
            "Users may replace this with a different implementation when subclassing the API."
        ),
    )

    # =======================================================================================================
    # === Pydantic validators ===============================================================================

    @pydantic.root_validator(pre=True)  # noqa
    @classmethod
    def _valid_tracking_uri(cls, values: dicts.AnyDict) -> dicts.AnyDict:
        """Use default if provided value is None or empty string."""
        raw_url = values.get("tracking_uri") or os.environ.get("MLFLOW_TRACKING_URI") or (Path.home() / ".cache/mlflow")
        values["tracking_uri"] = str(urls.parse_url(raw_url, resolve_if_local=True))
        return values

    def __init__(self, **kwargs):
        """Let the query push down use the same data translator as the API."""
        super().__init__(**kwargs)
        if self.query_push_down.data_translation is None:
            self.query_push_down.data_translation = self.data_translation

        if self.healthcheck and not self.offline_mode:
            with self._client() as cli:
                cli.healthcheck()

    # =======================================================================================================
    # === Properties ========================================================================================

    @property
    def _default_cache_id(self) -> str:
        """Sub-dir under default cache dir. Only used if `cache_dir` is not specified."""
        return hashlib.md5(self.tracking_uri.encode()).hexdigest()[:16]

    # =======================================================================================================
    # === Client ============================================================================================

    def _get_client(self) -> MlflowClient:
        assert not self.offline_mode, "Cannot use MlflowClient in offline mode."
        return MlflowClient(self.tracking_uri)

    @contextlib.contextmanager
    def _client(self) -> ContextManager[MlflowClient]:
        with env_utils.using_env_vars({"MLFLOW_%s" % k.upper(): str(v) for k, v in self.client_settings.items()}):
            yield self._get_client()

    def _using_client(self, func: Callable[[MlflowClient], A]) -> A:
        with self._client() as client:
            return func(client)

    # =======================================================================================================
    # === Metadata parsers ==================================================================================

    def _parse_experiment(self, native_experiment: NE) -> E:
        return E(
            name=native_experiment.name,
            id=native_experiment.experiment_id,
            tags=self.data_translation.deprocess_tags(native_experiment.tags),
        )

    def _parse_run(self, native_run: NR, experiment: E) -> R:
        if not (repo := native_run.data.tags.get(self.tag_keys.artifacts_repo)):
            repo = self._using_client(lambda client: client.get_artifact_uri(native_run.info.run_id))

        return R(
            exp=experiment,
            id=native_run.info.run_id,
            name=native_run.info.run_name,
            repo=str(urls.parse_url(repo, resolve_if_local=True)),
            tags=self.data_translation.deprocess_tags(native_run.data.tags),
            params=self.data_translation.deprocess_params(native_run.data.params),
            metrics=self.data_translation.deprocess_metrics(native_run.data.metrics),
            status=self.data_translation.run_status_from_mlflow(native_run.info.status),
            end_time=self.data_translation.mlflow_ts_to_datetime(native_run.info.end_time),
            start_time=self.data_translation.mlflow_ts_to_datetime(native_run.info.start_time),
        )

    def _parse_model(self, native_model: NM) -> M:
        return M(
            name=native_model.name,
            tags=self.data_translation.deprocess_tags(native_model.tags),
        )

    def _parse_model_version(self, native_mv: NV, model: M, run: R) -> V:
        path_in_run = (
            str(urls.parse_url(native_mv.source)).removeprefix(f"runs:///{run.id}/").removeprefix(run.repo).strip("/")
        )

        return V(
            run=run,
            model=model,
            path_in_run=path_in_run,
            version=str(native_mv.version),
            tags=self.data_translation.deprocess_tags(native_mv.tags),
        )

    # =======================================================================================================
    # === Implementations of abstract methods from `BaseMlflowApi` ==========================================

    def _impl_default_cache_dir(self) -> Path:
        """Get default cache dir based on the current MLflow API settings."""
        return Path.home().joinpath(".cache/mlopus/mlflow-providers/mlflow", self._default_cache_id)

    def _impl_get_exp_url(self, exp_id: str) -> urls.Url:
        """Get Experiment URL."""
        path = "" if urls.is_local(base := self.tracking_uri) else "#/experiments"
        return urls.urljoin(base, path, exp_id)

    def _impl_get_run_url(self, run_id: str, exp_id: str) -> urls.Url:
        """Get Run URL."""
        path = "" if urls.is_local(base := self._impl_get_exp_url(exp_id)) else "runs"
        return urls.urljoin(base, path, run_id)

    def _impl_get_model_url(self, name: str) -> urls.Url:
        """Get URL to registered model."""
        name = patterns.encode_model_name(name)
        path = "models/%s" if urls.is_local(base := self.tracking_uri) else "#/models/%s"
        return urls.urljoin(base, path % name)

    def _impl_get_mv_url(self, name: str, version: str) -> urls.Url:
        """Get model version URL."""
        name = patterns.encode_model_name(name)
        path = "models/%s/version-%s" if urls.is_local(base := self.tracking_uri) else "#/models/%s/versions/%s"
        return urls.urljoin(base, path % (name, version))

    def _impl_fetch_exp(self, exp_id: str) -> E:
        """Get Experiment by ID."""
        with self._client() as cli:
            return self._parse_experiment(cli.get_experiment(exp_id))

    def _impl_fetch_run(self, run_id: str) -> R:
        """Get Run by ID."""
        native_run = self._using_client(lambda client: client.get_run(run_id))
        experiment = self._fetch_exp(native_run.info.experiment_id)
        return self._parse_run(native_run, experiment)

    def _impl_fetch_model(self, name: str) -> M:
        """Get registered Model by name."""
        return self._parse_model(self._using_client(lambda client: client.get_registered_model(name)))

    def _impl_fetch_mv(self, name_and_version: Tuple[str, str]) -> V:
        """Get ModelVersion by name and version."""
        native_mv = self._using_client(lambda client: client.get_model_version(*name_and_version))
        run = self._fetch_run(native_mv.run_id)
        model = self._fetch_model(native_mv.name)
        return self._parse_model_version(native_mv, model, run)

    def _impl_find_experiments(
        self, query: mongo.Query, sorting: mongo.Sorting
    ) -> Tuple[mongo.Query, mongo.Sorting, iter_utils.Paginator[E]]:
        """Push down MongoDB query where possible and return query remainder with results iterator."""
        filter_expr, query_remainder, sort_expr, sort_remainder = self.query_push_down.parse_exp(query, sorting)
        logger.debug("Query push down (exp): %s %s", filter_expr, sort_expr)

        paginator = iter_utils.Paginator[NE](
            lambda token: _open_paged_list(
                self._using_client(
                    lambda client: client.search_experiments(
                        page_token=token,
                        filter_string=filter_expr,
                        order_by=sort_expr.split(", ") if sort_expr else None,
                    )
                ),
            ),
        )

        return query_remainder, sort_remainder, paginator.map_results(self._parse_experiment)

    def _impl_find_runs(
        self, query: mongo.Query, sorting: mongo.Sorting
    ) -> Tuple[mongo.Query, mongo.Sorting, iter_utils.Paginator[R]]:
        """Push down MongoDB query where possible and return query remainder with results iterator."""
        filter_expr, query_remainder, sort_expr, sort_remainder, exp_ids = self.query_push_down.parse_run(
            query, sorting
        )
        logger.debug("Query push down (run): %s %s", filter_expr, sort_expr)

        paginator = iter_utils.Paginator[NR](
            lambda token: _open_paged_list(
                self._using_client(
                    lambda client: client.search_runs(
                        page_token=token,
                        experiment_ids=exp_ids,
                        filter_string=filter_expr,
                        order_by=sort_expr.split(", ") if sort_expr else None,
                    )
                ),
            ),
        ).map_results(
            lambda native_run: self._parse_run(
                native_run=native_run,
                experiment=self._fetch_exp(native_run.info.experiment_id),
            ),
        )

        return query_remainder, sort_remainder, paginator

    def _impl_find_models(
        self, query: mongo.Query, sorting: mongo.Sorting
    ) -> Tuple[mongo.Query, mongo.Sorting, iter_utils.Paginator[M]]:
        """Push down MongoDB query where possible and return query remainder with results iterator."""
        filter_expr, query_remainder, sort_expr, sort_remainder = self.query_push_down.parse_model(query, sorting)
        logger.debug("Query push down (model): %s %s", filter_expr, sort_expr)

        paginator = iter_utils.Paginator[NM](
            lambda token: _open_paged_list(
                self._using_client(
                    lambda client: client.search_registered_models(
                        page_token=token,
                        filter_string=filter_expr,
                        order_by=sort_expr.split(", ") if sort_expr else None,
                    )
                ),
            ),
        )

        return query_remainder, sort_remainder, paginator.map_results(self._parse_model)

    def _impl_find_mv(
        self, query: mongo.Query, sorting: mongo.Sorting
    ) -> Tuple[mongo.Query, mongo.Sorting, iter_utils.Paginator[V]]:
        """Push down MongoDB query where possible and return query remainder with results iterator."""
        filter_expr, query_remainder, sort_expr, sort_remainder = self.query_push_down.parse_mv(query, sorting)
        logger.debug("Query push down (mv): %s %s", filter_expr, sort_expr)

        paginator = iter_utils.Paginator[NV](
            lambda token: _open_paged_list(
                self._using_client(
                    lambda client: client.search_model_versions(
                        page_token=token,
                        filter_string=filter_expr,
                        order_by=sort_expr.split(", ") if sort_expr else None,
                    )
                ),
            ),
        ).map_results(
            lambda native_mv: self._parse_model_version(
                native_mv=native_mv,
                run=self._fetch_run(native_mv.run_id),
                model=self._fetch_model(native_mv.name),
            ),
        )

        return query_remainder, sort_remainder, paginator

    def _impl_find_child_runs(self, run: R) -> Iterator[R]:
        """Find child runs."""
        return self.find_runs({"exp.id": run.exp.id, f"tags.{self.tag_keys.parent_run_id}": run.id})

    def _impl_create_exp(self, name: str, tags: Mapping) -> E:
        """Create experiment and return its metadata."""
        tags = self.data_translation.preprocess_tags(tags)
        return self._get_exp(self._using_client(lambda client: client.create_experiment(name=name, tags=tags)))

    def _impl_create_model(self, name: str, tags: Mapping) -> M:
        """Create registered model and return its metadata."""
        tags = self.data_translation.preprocess_tags(tags)
        return self._parse_model(self._using_client(lambda client: client.create_registered_model(name, tags)))

    def _impl_create_run(
        self, exp_id: str, name: str | None, repo: urls.Url | None, parent_run_id: str | None = None
    ) -> str:
        """Create run."""
        with self._client() as client:
            run_id = client.create_run(exp_id, run_name=name).info.run_id
            if repo:
                client.set_tag(run_id, key=self.tag_keys.artifacts_repo, value=str(repo), synchronous=True)
            if parent_run_id:
                client.set_tag(run_id, key=self.tag_keys.parent_run_id, value=parent_run_id, synchronous=True)
            return run_id

    def _impl_set_run_status(self, run_id: str, status: schema.RunStatus):
        """Set Run status."""
        with self._client() as client:
            client.update_run(run_id, status=self.data_translation.run_status_to_mlflow(status, as_str=True))

    def _impl_set_run_end_time(self, run_id: str, end_time: datetime):
        """Set Run end time."""
        end_time = self.data_translation.datetime_to_mlflow_ts(end_time)
        self._using_client(lambda client: client.tracking_client.store.update_run_info(run_id, None, end_time, None))

    def _impl_update_exp_tags(self, exp_id: str, tags: Mapping):
        """Update Exp tags."""
        with self._client() as client:
            for k, v in self.data_translation.preprocess_tags(tags).items():
                client.set_experiment_tag(exp_id, k, v)

    def _impl_update_run_tags(self, run_id: str, tags: Mapping):
        """Update Run tags."""
        with self._client() as client:
            for k, v in self.data_translation.preprocess_tags(tags).items():
                client.set_tag(run_id, k, v, synchronous=True)

    def _impl_update_model_tags(self, name: str, tags: Mapping):
        """Update Model tags."""
        with self._client() as client:
            for k, v in self.data_translation.preprocess_tags(tags).items():
                client.set_registered_model_tag(name, k, v)

    def _impl_update_mv_tags(self, name: str, version: str, tags: Mapping):
        """Update Exp tags."""
        with self._client() as client:
            for k, v in self.data_translation.preprocess_tags(tags).items():
                client.set_model_version_tag(name, version, k, v)

    def _impl_log_run_params(self, run_id: str, params: Mapping):
        """Log run params."""
        with self._client() as client:
            for k, v in self.data_translation.preprocess_params(params).items():
                client.log_param(run_id, k, v, synchronous=True)

    def _impl_log_run_metrics(self, run_id: str, metrics: Mapping):
        """Log run metrics."""
        with self._client() as client:
            for k, v in self.data_translation.preprocess_metrics(metrics).items():
                client.log_metric(run_id, k, v, synchronous=True)

    def _impl_register_mv(self, model: M, run: R, path_in_run: str, version: str | None, tags: Mapping) -> V:
        """Register model version."""
        assert version is None, f"Arbitrary `version` not supported in '{self.__class__.__name__}'"
        tags = self.data_translation.preprocess_tags(tags)
        source = str(urls.urljoin(run.repo, path_in_run))
        native_mv = self._using_client(lambda client: client.create_model_version(model.name, source, run.id, tags))
        return self._parse_model_version(native_mv, model, run)


def _open_paged_list(paged_list: PagedList[A]) -> iter_utils.Page[A]:
    """Convert MLflow native PagedList to `iter_utils.Page`."""
    return iter_utils.Page(token=paged_list.token, results=paged_list)
