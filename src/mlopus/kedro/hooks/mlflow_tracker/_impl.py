import contextlib
import logging
from pathlib import Path
from typing import Any, List

from kedro.framework.context import KedroContext
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.pipeline import Pipeline

import mlopus
from mlopus.utils import pydantic, yaml, dicts, import_utils
from ._pipeline import _Pipeline, _Node
from ._state import _State
from ._tracker import Logs, Report, Tags, Metrics, Params, Overrides, Config, Nodes, Datasets
from ..hook_factory import HookWithFactory  # noqa: TID252
from ...utils import log_error  # noqa: TID252

logger = logging.getLogger(__name__)


class MlflowTracker(mlopus.mlflow.MlflowRunMixin, HookWithFactory):
    """Hook to track pipeline information in MLflow.

    Find `here <https://github.com/lariel-fernandes/mlopus/tree/main/examples/2_a_kedro_project/conf/full/parameters/hooks/mlflow_tracker.yml>`_
    a fully commented example covering all settings that can be customized in this hook.
    """

    logs: Logs = pydantic.Field(
        default=Logs(enabled=True),
        description="Configure how to track logs.",
    )
    report: Report = pydantic.Field(
        default=Report(enabled=True),
        description="Configure the session report.",
    )
    tags: Tags = pydantic.Field(
        default=Tags(report=False),
        description="Configure extra tags.",
    )
    metrics: Metrics = pydantic.Field(
        default=Metrics(report=False),
        description="Configure the collection of MLflow metrics.",
    )
    params: Params = pydantic.Field(
        default=Params(enable=False),
        description="Configure the collection of MLflow params.",
    )
    overrides: Overrides = pydantic.Field(
        default=Overrides(report=False),
        description="Configure how to track Kedro config overrides.",
    )
    config: Config = pydantic.Field(
        default=Config(report=False),
        description="Configure how to track the Kedro config.",
    )
    nodes: Nodes = pydantic.Field(
        default=Nodes(report=True),
        description="Configure how to track Kedro node settings.",
    )
    datasets: Datasets = pydantic.Field(
        default=Datasets(report=True),
        description="Configure how to track Kedro dataset settings.",
    )
    state: _State = pydantic.Field(exclude=True, default_factory=_State)

    def __init__(self, **kwargs):
        assert "state" not in kwargs, "Field `state` can only be initialized from defaults."
        super().__init__(**kwargs)
        self.state.report.tags = self.tags.values  # Propagate tags to report state

    # =======================================================================================================
    # === Checkpoint recovery Methods =======================================================================

    def _load_reports(self) -> List[dict]:
        """Load existing session reports from a previous session report file.

        Useful in case this run has been resumed by ID, so this session's report can be added appended.
        """
        try:
            return self.run_manager.run.load_artifact(
                path_in_run=self.report.path,
                loader=lambda path: yaml.loads(path.read_text()),
            )
        except FileNotFoundError:
            return []

    def _load_logs(self, path_in_run: str) -> str:
        """Load existing logs from a previous session log file.

        Useful in case this run has been resumed by ID, so this session's log can be added appended.
        """
        try:
            content = self.run_manager.run.load_artifact(
                path_in_run=path_in_run,
                loader=lambda path: path.read_text(),
            )
            return (content or "(empty)\n") + "\n# " + ("=" * 98) + "\n\n"  # footer line
        except FileNotFoundError:
            return ""

    # =======================================================================================================
    # === Tracker Methods ===================================================================================

    def _track_context(self, context: KedroContext):
        self.state.conf = context.config_loader
        self.state.report.config = self.config.apply(context.config_loader)
        self.state.report.overrides = self.overrides.apply(context.config_loader.runtime_params)

    def _track_catalog(self, catalog: DataCatalog):
        self.state.datasets = catalog._datasets  # noqa

    def _track_dataset(self, dataset_name: str):
        if (ds := self.state.datasets.get(dataset_name)) is None:
            return

        ds_info = {"type": import_utils.fq_name(type(ds))}

        if pydantic_ds := pydantic.as_model_obj(ds):
            ds_info["conf"] = pydantic_ds.dict()
        elif self.datasets.include_non_pydantic:
            ds_info["conf"] = ds.__dict__

        if self.datasets.allow((dataset_name, ds_info)):
            self.state.report.datasets[dataset_name] = ds_info

    def _track_metrics(self, dataset_name: str, data: Any):
        if dataset_name in self.metrics.datasets:
            logger.info("Collecting metrics from dataset '%s'", dataset_name)
            if (metrics := self.state.report.metrics.get(dataset_name)) is not None:
                if not isinstance(metrics, list):
                    metrics = [metrics]
                metrics.append(data)
            else:
                metrics = data

            self.state.report.metrics[dataset_name] = metrics

    def _track_pipeline(self, pipeline_name: str, pipeline: Pipeline, **pipeline_args):
        self.state.report.params.update(self.params.apply(pipeline_name, self.state.conf))
        self.state.report.pipelines[pipeline_name] = _Pipeline.parse_obj(pipeline_args)
        for node in pipeline.nodes:
            if not self.nodes.allow(_node := _Node.from_kedro(node)) or not self.nodes.report:
                _node.func = None
            self.state.report.pipelines[pipeline_name].nodes[node.name] = _node

    def _track_pipeline_outcome(self, pipeline_name: str, error: Exception | None):
        self.state.report.pipelines[pipeline_name].set_error(error)

    # =======================================================================================================
    # === Data rendering ====================================================================================

    def _render_report(self) -> List[dict]:
        """Render report dict based on collected data and tracker settings."""
        report = self.state.report.dict(
            exclude={
                field: not getattr(self, field).report
                for field in ["tags", "metrics", "params", "config", "overrides", "datasets"]
            },
        )
        return self._load_reports() + [report]

    def _render_tags(self) -> dict | None:
        """Render MLflow tags based on report state and tracker settings."""
        return self.tags.mlflow.apply(self.state.report.tags) if self.tags.mlflow.enabled else None

    def _render_metrics(self) -> dict | None:
        """Render MLflow metrics based on collected data and tracker settings."""
        return self.metrics.mlflow.apply(self.state.report.metrics) if self.metrics.mlflow.enabled else None

    def _render_params(self) -> dict:
        """Render MLflow params based on collected data and tracker settings."""
        return dicts.deep_merge(
            self.params.mlflow.apply(self.state.report.params) if self.params.mlflow.enabled else {},
            self.datasets.mlflow.apply(self.state.report.datasets) if self.datasets.mlflow.enabled else {},
            self.config.mlflow.apply(self.state.report.config) if self.config.mlflow.enabled else {},
            self.overrides.mlflow.apply(self.state.report.overrides) if self.overrides.mlflow.enabled else {},
            self.nodes.mlflow.apply(self.state.report.pipelines) if self.nodes.mlflow.enabled else {},
        )

    # =======================================================================================================
    # === Data serialization ================================================================================

    def _dump_report(self) -> str:
        return yaml.dumps(self._render_report())

    # =======================================================================================================
    # === Data persistence ==================================================================================

    def _clean_logs(self) -> None:
        if self.logs.enabled:
            [path.write_text("") for file in self.logs.files if file.cleanup and (path := Path(file.path)).exists()]

    def _save_report(self, path: Path) -> None:
        path.write_text(self._dump_report())

    # =======================================================================================================
    # === Data transfer =====================================================================================

    def _log_report(self, force: bool = False):
        if self.report.enabled or force:
            self.run_manager.run.log_artifact(self._save_report, self.report.path, use_cache=False)

    def _set_tags(self):
        if tags := self._render_tags():
            self.run_manager.run.set_tags(tags)

    def _log_metrics(self):
        if metrics := self._render_metrics():
            self.run_manager.run.log_metrics(metrics)

    def _log_params(self):
        if params := self._render_params():
            self.run_manager.run.log_params(params)

    def _log_logs(self):
        if not self.logs.enabled:
            return

        dir_in_run = Path(self.logs.path)

        for file_conf in self.logs.files:
            if not (log_file := Path(file_conf.path)).is_file():
                continue

            self.run_manager.run.log_artifact(
                use_cache=False,
                path_in_run=(path_in_run := str(dir_in_run / (file_conf.alias or log_file.name))),
                source=lambda tmp: tmp.write_text(self._load_logs(path_in_run) + log_file.read_text()),
            )

    # =======================================================================================================
    # === Experiment flow ===================================================================================

    def _resume_run(self):
        self.run_manager.run.resume()

    def _end_run(self, succeeded: bool = True):
        self.run_manager.run.end_run(succeeded=succeeded)

    # =======================================================================================================
    # === Hook logic ========================================================================================

    @contextlib.contextmanager
    def _ensure_logs(self):
        """Context manager for hook methods that may fail. Guarantees that errors are logged and logs are uploaded."""
        try:
            yield None
        except Exception as error:
            log_error(error, logger)  # If there was an error while tracking, make sure it goes to logs before uploading
            self._log_logs()  # Upload logs
            raise error

    def _before_pipeline(self, run_params: dict, pipeline: Pipeline):
        if len(self.state.report.pipelines) == 0:
            self._set_tags()  # delay setting tags until the first pipeline is about to be run
        self._resume_run()
        self._track_pipeline(**run_params, pipeline=pipeline)

    def _after_pipeline(self, run_params: dict, error: Exception | None = None):
        self._end_run(succeeded=error is None)
        self._track_pipeline_outcome(run_params["pipeline_name"], error=error)
        self._log_report()
        self._log_params()
        self._log_metrics()
        log_error(error, logger)  # If there was an error in the pipeline, make sure it goes to logs before the upload
        self._log_logs()  # Upload logs

    # =======================================================================================================
    # === Hook triggers =====================================================================================

    @hook_impl
    def after_context_created(self, context: KedroContext):
        self._clean_logs()
        with self._ensure_logs():
            self._track_context(context)

    @hook_impl
    def after_catalog_created(
        self,
        catalog: DataCatalog,
        # conf_catalog: dict[str, Any],
        # conf_creds: dict[str, Any],
        # feed_dict: dict[str, Any],
        # save_version: str,
        # load_versions: dict[str, str],
    ):
        with self._ensure_logs():
            self._track_catalog(catalog)

    @hook_impl
    def before_dataset_saved(
        self,
        dataset_name: str,
        data: Any,
        # node: Node
    ):
        with self._ensure_logs():
            self._track_dataset(dataset_name)
            self._track_metrics(dataset_name, data)

    @hook_impl
    def before_dataset_loaded(
        self,
        dataset_name: str,
        # node: Node
    ):
        with self._ensure_logs():
            self._track_dataset(dataset_name)

    @hook_impl
    def before_pipeline_run(
        self,
        run_params: dict,
        pipeline: Pipeline,
        # catalog: DataCatalog,
    ):
        with self._ensure_logs():
            self._before_pipeline(run_params, pipeline)

    @hook_impl
    def after_pipeline_run(
        self,
        run_params: dict,
        # run_result: dict,
        # pipeline: Pipeline,
        # catalog: DataCatalog,
    ):
        with self._ensure_logs():
            self._after_pipeline(run_params)

    @hook_impl
    def on_pipeline_error(
        self,
        error: Exception,
        run_params: dict,
        # pipeline: Pipeline,
        # catalog: DataCatalog,
    ):
        with self._ensure_logs():
            self._after_pipeline(run_params, error=error)
