import copy
from abc import ABC, abstractmethod
from typing import Callable, List, Dict, Any, Type, Iterable

import click
from kedro.framework.cli.project import run as kedro_run_command
from kedro.framework.session import KedroSession

from mlopus.utils import pydantic, dicts, func_utils, logical


class _Callback(pydantic.BaseModel, ABC):
    """Base class for callbacks to be used with a `WrappedKedroRunCommand`."""

    callback: Callable[[...], Any]
    pipelines: List[str] | None = None

    def enabled_for(self, pipeline: str) -> bool:
        """Tell if this callback is enabled for the specified pipeline."""
        return not self.pipelines or pipeline in self.pipelines

    def _run_callback(self, **kwargs) -> Any:
        original = copy.deepcopy(kwargs)

        retval = func_utils.call_with_kwargs(self.callback, kwargs)

        if kwargs != original:
            raise RuntimeError("The callback must not modify its arguments!")

        return retval

    @abstractmethod
    def __call__(self, **kwargs) -> None:
        pass


class _SideEffect(_Callback):
    """A callback that produces side effects only."""

    callback: Callable[[...], None]

    def __call__(self, **kwargs) -> None:
        self._run_callback(**kwargs)


class _DynamicOverride(_Callback):
    """A callback that injects a value in the `--params` option of `kedro run` (aka. runtime params or overrides)."""

    target_key: str
    callback: Callable[[...], str | int | float | dict | list | None]

    def __call__(self, **kwargs) -> None:
        if dicts.has_nested(params := kwargs["cli_args"]["params"], key := self.target_key.split(".")):
            return  # If user has provided an override for this key already, leave it as it is

        if (val := self._run_callback(**kwargs)) is not None:  # if callback returns None, abort override
            dicts.set_nested(params, key, val)


class _DynamicRootOverride(_Callback):
    """A callback that injects a dict of values in the root of the `--params` option of `kedro run`."""

    callback: Callable[[...], str | int | float | dict | list | None]

    def __call__(self, **kwargs) -> None:
        kwargs["cli_args"]["params"] = dicts.deep_merge(
            self._run_callback(**kwargs) or {},
            kwargs["cli_args"]["params"],
        )


class RunCommand(pydantic.BaseModel):
    """Wrapper around the `kedro run` command."""

    name: str = "run"
    help: str | None = None
    command: click.Command = None
    callbacks: List[_Callback] = []
    decorators: List[Callable[[callable], callable]] = []

    @pydantic.root_validator(allow_reuse=True, pre=True)  # noqa
    @classmethod
    def _validate_command(cls, values: dict) -> dict:
        values["command"] = copy.deepcopy(values.get("command") or kedro_run_command)
        values["command"].context_settings["ignore_unknown_options"] = True
        return values

    @pydantic.root_validator(allow_reuse=True, pre=True)  # noqa
    @classmethod
    def _validate_decorators(cls, values: dict) -> dict:
        values.setdefault("decorators", []).append(click.pass_context)
        return values

    def register(self, cli: click.Group) -> "RunCommand":
        """Register the wrapped `kedro run` command in the specified CLI."""
        decorate = func_utils.compose(self.decorators)
        func = decorate(lambda ctx, **kw: self(ctx, **kw))
        cmd = cli.command(name=self.name, help=self.help or self.command.help)(func)
        cmd.params += self.command.params

        n_callbacks = len(self.callbacks)  # number of custom callbacks

        for opt in cmd.params:  # evaluate callbacks from CLI options
            if isinstance(opt, Option):
                opt.forward_override(self)

        # send custom callbacks to the end and callbacks from CLI options to the beginning
        self.callbacks = self.callbacks[n_callbacks:] + self.callbacks[:n_callbacks]

        return self

    def side_effect(self, pipelines: List[str] | None = None):
        """Decorate function to produce a side effect before running the pipeline.

        Used in all pipelines, unless a list of pipelines is specified.
        The function must return None and must not change its arguments.
        The function may expect any of the following arguments: cli_args, kedro_config.
        """

        def decorator(func: callable):
            self._add_callback(_SideEffect, pipelines=pipelines, callback=func)

        return decorator

    def dynamic_override(self, target_key: str, pipelines: List[str] | None = None):
        """Decorate function to inject a runtime param (config override) before running the pipeline.

        Used in all pipelines, unless a list of pipelines is specified.
        The function must return the new param value and must not change its arguments.
        The function may expect any of the following arguments: cli_args, kedro_config.
        """

        def decorator(func: callable):
            self._add_callback(_DynamicOverride, pipelines=pipelines, callback=func, target_key=target_key)

        return decorator

    def dynamic_root_override(self, pipelines: List[str] | None = None):
        """Decorate function to inject runtime params (config overrides) before running the pipeline.

        Used in all pipelines, unless a list of pipelines is specified.
        The function must return a dict of params and must not change its arguments.
        The function may expect any of the following arguments: cli_args, kedro_config.
        """

        def decorator(func: callable):
            self._add_callback(_DynamicRootOverride, pipelines=pipelines, callback=func)

        return decorator

    def __call__(self, ctx: click.Context, **__):
        for cb in self._pipeline_callbacks(ctx.params["pipeline"]):
            with KedroSession.create(
                env=ctx.params["env"],
                extra_params=ctx.params["params"],
                conf_source=ctx.params["conf_source"],
            ) as session:
                cb(**self._build_callback_kwargs(ctx, session))

        ctx.params = func_utils.adjust_kwargs(self.command.callback, ctx.params)
        ctx.forward(self.command)

    def _pipeline_callbacks(self, pipeline: str) -> Iterable[_Callback]:
        return (cb for cb in self.callbacks if cb.enabled_for(pipeline))

    @classmethod
    def _build_callback_kwargs(cls, ctx: click.Context, session: KedroSession) -> Dict[str, Any]:
        return {
            "cli_args": ctx.params,
            "kedro_config": session.load_context().config_loader,
        }

    def _add_callback(self, cls: Type[_Callback], **kwargs):
        self.callbacks.append(cls(**kwargs))


class Option(click.Option):
    """CLI option with extra attributes.

    :param params_root: If True, a dict passed to this CLI option will be forwarded to the root of the Kedro
                       runtime params (overrides). Incompatible with `target_key`.

    :param target_key: If specified, the value passed to this CLI option will be forwarded to the Kedro
                       runtime params (overrides) at this key path (e.g.: globals.model.name).

    :param pipelines: (Optional) Restrict the usage of `target_key` to the specified pipelines.
    """

    def __init__(
        self,
        *args,
        params_root: bool = False,
        target_key: str | None = None,
        pipelines: List[str] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._pipelines = pipelines
        self._target_key = target_key
        self._params_root = params_root
        assert logical.xor(target_key, params_root), "`target_key` and `params_root` are incompatible."

    def _get_value(self, cli_args: dict) -> Any:
        return cli_args.get(self.name)

    def forward_override(self, cmd: RunCommand):
        """Add dynamic override callback to run command based on this CLI option."""
        if self._params_root:
            cmd.dynamic_root_override(self._pipelines)(self._get_value)
        elif self._target_key:
            cmd.dynamic_override(self._target_key, self._pipelines)(self._get_value)


FC = Callable[[...], Any] | click.Command


def cli_option(*param_decls, **attrs) -> Callable[[FC], FC]:
    """CLI option with class `mlopus.kedro.cli_tools.Option`."""
    return click.option(*param_decls, cls=Option, **attrs)
