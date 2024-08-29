import contextlib
import logging
import os
import uuid
from pathlib import Path
from typing import Iterable, Any

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
from .hook_factory import HookFactory
from .pipeline_factory import PipelineFactory
from .utils import log_errors

logger = logging.getLogger(__name__)


class MlopusKedroSession(KedroSession):
    """Patch of KedroSession.

    Features:
      - Exposes the session store data for interpolation via config keys.
        (e.g.: in `parameters.yml`, my_key: ${session:session_id})

      - Exposes environment variables for interpolation via config keys in any scope.
        (e.g.: in `parameters.yml`, my_key: ${env:my_env_var})

      - Allows pipelines to be built dynamically by functions that receive the Kedro config and return `Pipeline`
        Example:
            # in `pipeline_registry.py`, where `create_pipeline` expects `config` and returns `Pipeline`
            "my_pipeline": pipeline_factory(create_pipeline)

      - Allows hooks to be built dynamically by functions that receive the Kedro config and return a hook
        Example:
            # in `settings.py`, where `create_hook` expects `config` and returns a hook object
            HOOKS = [hook_factory(create_hook), ...]
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

        self._store["pkg"] = {"name": self._package_name, "version": packaging.get_dist(self._package_name).version}

        self._store["uuid"] = self.uuid = str(uuid.uuid4())  # generate UUID (not datetime-bound like the session ID)

        self._hook_manager.trace.root.setwriter(None)  # fix to prevent data dumping on call to pluggy

        DictResolver(self._store).register("session")  # expose session info for interpolation in config keys

        DictResolver(os.environ).register("env")  # expose environment variables for interpolation in config keys

        self._ctx = None  # lazy initialized cached context

    def create_context(self) -> KedroContext:
        """Load and cache context. Initialize and register hooks now that config is available."""
        ctx = super().load_context()  # evaluate the context
        self._store["env"] = ctx.config_loader.env  # save env name (default is only applied on context creation)
        hooks = [self._load_hook(ctx.config_loader, hook) for hook in settings.HOOKS]  # evaluate hook factories
        _register_hooks(self._hook_manager, hooks)  # register hooks
        self._hook_manager.hook.after_context_created(context=ctx)  # run post-context creation hooks
        return ctx

    def load_context(self) -> KedroContext:
        """Get cached context."""
        if self._ctx is None:
            self._ctx = self.create_context()
        return self._ctx

    @log_errors(logger)
    def run(  # noqa: PLR0913
        self,
        pipeline_name: str | None = None,
        tags: Iterable[str] | None = None,
        runner: AbstractRunner | None = None,
        node_names: Iterable[str] | None = None,
        from_nodes: Iterable[str] | None = None,
        to_nodes: Iterable[str] | None = None,
        from_inputs: Iterable[str] | None = None,
        to_outputs: Iterable[str] | None = None,
        load_versions: dict[str, str] | None = None,
        namespace: str | None = None,
    ) -> dict[str, Any]:
        with self._loaded_pipeline(self.load_context().config_loader, pipeline_name := pipeline_name or "__default__"):
            return super().run(
                pipeline_name,
                tags,
                runner,
                node_names,
                from_nodes,
                to_nodes,
                from_inputs,
                to_outputs,
                load_versions,
                namespace,
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
