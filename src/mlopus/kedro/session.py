import contextlib
import inspect
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterable, Any
from kedro import __version__
import importlib_metadata
import pytz
import toml
from kedro.config import AbstractConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.hooks.manager import _register_hooks  # noqa
from kedro.framework.project import pipelines, settings
from kedro.framework.session import KedroSession
from kedro.pipeline import Pipeline
from kedro.runner import AbstractRunner

from mlopus.utils import pydantic, packaging
from .config_resolvers import DictResolver
from .hooks import HookFactory
from .pipeline_factory import PipelineFactory
from .utils import log_errors

logger = logging.getLogger(__name__)

_create_session = KedroSession.create  # Save reference to non-patched session factory

_kedro_major = int(__version__.split(".")[0])  # Get major version for backwards compatibility handling


class MlopusKedroSession(KedroSession):
    """Patch of KedroSession.

    Enabling the patch
    ==================

        .. code-block:: python

            # <your_package>/settings.py
            from mlopus.kedro import MlopusKedroSession
            from kedro.framework.session import KedroSession

            KedroSession.create = MlopusKedroSession.create

    Resolving env vars and session store details in config files
    ============================================================

        .. code-block:: yaml

            # conf/<env>/parameters.yml
            my_env_var: "${env:MY_ENV_VAR,default}"  # resolve env var
            package_version: "${session:pkg.version}"  # resolve session store details

    Lazy-evaluated pipelines with direct config access
    ==================================================

    In the following example, the function `prepare_images` will be called to build the
    pipeline from the config **only** when the respective pipeline is chosen for execution.

    If the node function `SetImageContrast` is a Pydantic BaseModel or has any other form of
    schema validation, the mapped configuration will be validated **before** the pipeline runs.

        .. code-block:: python

            # <your_package>/pipeline_registry.py
            from mlopus.kedro import pipeline_factory

            def register_pipelines():
                return {"e2e": prepare_images}

            @pipeline_factory
            def prepare_images(config):
                return Pipeline([
                    node(
                        name="set_contrast",
                        inputs="original_images",
                        outputs="modified_images",
                        func=SetImageContrast(config["parameters"]["contrast"]),
                    ),
                ])

    Lazy-evaluated hooks with direct config access
    ==============================================

        .. code-block:: python

            # <your_package>/settings.py
            from mlopus.kedro import hook_factory

            @hook_factory
            def upload_logs(config):
                return UploadLogs(bucket=config["globals"]["logs_bucket"])

            HOOKS = [upload_logs]
    """

    def __init__(
        self,
        session_id: str,
        package_name: str | None = None,
        project_path: Path | str | None = None,
        save_on_close: bool = False,
        conf_source: str | None = None,
    ):
        with self._hiding_hooks():  # prevent parent class from registering uninitialized hooks
            super().__init__(session_id, package_name, project_path, save_on_close, conf_source)

        if self._package_name is None:  # resolve package name from project metadata if not specified
            self._package_name = toml.load(self._project_path / "pyproject.toml")["tool"]["kedro"]["package_name"]

        if not getattr(self, "_project_name", None):  # resolve project name from project metadata if not specified
            self._project_name = toml.load(self._project_path / "pyproject.toml")["tool"]["kedro"]["project_name"]

        try:
            dist = packaging.get_dist(self._package_name.split(".")[0])
        except importlib_metadata.PackageNotFoundError:
            # Fallback: look up the distribution metadata by project name instead
            # (useful in case the package name doesn't match the dist name exactly)
            dist = packaging.get_dist(self._project_name)

        self._store["dist"] = {
            "name": dist.name,
            "version": dist.version,
        }

        self._store["pkg"] = {
            "name": self._package_name,
            "version": dist.version,
        }

        self._store["timestamp"] = {
            "iso": (session_datetime := datetime.now(pytz.utc)).isoformat(),
            "unix": session_datetime.timestamp(),
        }  # include session start datetime both as ISO UTC and UNIX timestamp

        self._store["uuid"] = self.uuid = str(uuid.uuid4())  # generate UUID (not datetime-bound like the session ID)

        self._hook_manager.trace.root.setwriter(None)  # fix to prevent data dumping on call to pluggy

        DictResolver(self._store).register("session")  # expose session info for interpolation in config keys

        DictResolver(os.environ).register("env")  # expose environment variables for interpolation in config keys

        if "extra_namespaces" in inspect.signature(settings.CONFIG_LOADER_CLASS.__init__).parameters:
            settings.CONFIG_LOADER_ARGS = settings.CONFIG_LOADER_ARGS or {}
            settings.CONFIG_LOADER_ARGS["extra_namespaces"] = (
                settings.CONFIG_LOADER_ARGS.get("extra_namespaces") or {}
            ) | {"session": self._store}

        self._ctx = None  # lazy initialized cached context

    @classmethod
    def create(
        cls,
        project_path: Path | str | None = None,
        save_on_close: bool = True,
        env: str | None = None,
        extra_params: dict[str, Any] | None = None,
        runtime_params: dict[str, Any] | None = None,
        conf_source: str | None = None,
        **kwargs,
    ) -> KedroSession:
        """Patch of KedroSession.create offering backwards compatibility for the argument `extra_params`,
        which has been renamed to `runtime_params` in Kedro 1.0."""

        assert not (extra_params and runtime_params), "Specify at most one: `extra_params` or `runtime_params`"

        return _create_session.__func__(  # noqa
            cls,
            project_path=project_path,
            save_on_close=save_on_close,
            env=env,
            conf_source=conf_source,
            **{"extra_params" if _kedro_major == 0 else "runtime_params": extra_params or runtime_params},
        )

    def create_context(self) -> KedroContext:
        """Load and cache context. Initialize and register hooks now that config is available."""
        config_loader = self._get_config_loader()  # build config before evaluating hook factories
        self._store["env"] = config_loader.env  # save env name so super().load_context() uses the right env

        hooks = [self._load_hook(config_loader, hook) for hook in settings.HOOKS]  # evaluate hook factories with conf
        _register_hooks(self._hook_manager, hooks)  # register hooks before after_context_created fires

        self._ctx = super().load_context()  # create context; kedro v1+: fires context creation hooks

        if _kedro_major == 0:  # manually fire context creation hooks
            self._hook_manager.hook.after_context_created(context=self._ctx)

        return self._ctx

    def load_context(self) -> KedroContext:
        """Get cached context."""
        if self._ctx is None:
            self._ctx = self.create_context()
        return self._ctx

    @log_errors(logger)
    def run(  # noqa: PLR0913
        self,
        pipeline_name: str | None = None,
        pipeline_names: list[str] | None = None,
        tags: Iterable[str] | None = None,
        runner: AbstractRunner | None = None,
        node_names: Iterable[str] | None = None,
        from_nodes: Iterable[str] | None = None,
        to_nodes: Iterable[str] | None = None,
        from_inputs: Iterable[str] | None = None,
        to_outputs: Iterable[str] | None = None,
        load_versions: dict[str, str] | None = None,
        namespace: str | None = None,
        namespaces: list[str] | None = None,
        only_missing_outputs: bool | None = None,
    ) -> dict[str, Any]:
        """Patch of KedroSession.run with lazy pipeline evaluation and backwards compatibility for changed arguments."""
        kwargs = {}

        # Handle backwards compat for pipeline name
        assert not (pipeline_names and pipeline_name), "Specify at most one of `pipeline_name` or `pipeline_names`"
        if isinstance(pipeline_names, list) and len(pipeline_names) > 0:
            assert len(pipeline_names) == 1, "Multiple pipeline names are not supported"
            pipeline_name = pipeline_names[0]
        pipeline_name = pipeline_name or "__default__"

        # Handle backwards compat for namespace
        assert not (namespaces and namespace), "Specify at most one of `namespace` or `namespaces`"
        if isinstance(namespaces, list) and len(namespaces) > 0:
            assert len(namespaces) == 1, "Multiple namespaces are not supported"
            namespace = namespaces[0]
        if namespace:
            if _kedro_major == 0:
                kwargs["namespace"] = namespace
            else:
                kwargs["namespaces"] = [namespace]

        # Handle only_missing_outputs argument (added in Kedro 1.0)
        if only_missing_outputs is not None:
            assert _kedro_major >= 1, "The argument `only_missing_outputs` is only supported in Kedro 1.0+"
            kwargs["only_missing_outputs"] = only_missing_outputs

        with self._loaded_pipeline(self.load_context().config_loader, pipeline_name):
            return super().run(
                pipeline_name=pipeline_name,
                tags=tags,
                runner=runner,
                node_names=node_names,
                from_nodes=from_nodes,
                to_nodes=to_nodes,
                from_inputs=from_inputs,
                to_outputs=to_outputs,
                load_versions=load_versions,
                **kwargs,  # noqa
            )

    @classmethod
    @contextlib.contextmanager
    def _hiding_hooks(cls):
        hooks = settings.HOOKS  # save original hooks
        settings.HOOKS = []  # hide hooks
        yield None  # let the context run
        settings.HOOKS = hooks  # restore hooks

    @contextlib.contextmanager
    def _loaded_pipeline(self, config: AbstractConfigLoader, name: str):
        pipeline = pipelines[name]  # save original pipeline definition
        pipelines[name] = self._load_pipeline(name, config, pipeline)  # replace definition with loaded pipeline
        yield None  # let the context run
        pipelines[name] = pipeline  # restore the original pipeline definition

    @classmethod
    def _load_hook(cls, config: AbstractConfigLoader, hook: Any | HookFactory) -> Any:
        if isinstance(hook, HookFactory):
            hook = hook(config)
        return hook

    @classmethod
    @pydantic.validate_arguments(config={"arbitrary_types_allowed": True})
    def _load_pipeline(cls, name: str, config: AbstractConfigLoader, pipeline: Pipeline | PipelineFactory) -> Pipeline:
        if isinstance(pipeline, PipelineFactory):
            logger.debug("Loading pipeline '%s'...", name)
            pipeline = pipeline(config)
            logger.debug("Pipeline '%s' has been loaded", name)
        return pipeline
