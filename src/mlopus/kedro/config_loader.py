import typing
from typing import Iterable, Any

from kedro.config import MissingConfigException
from kedro.config.omegaconf_config import OmegaConfigLoader

from mlopus.utils import dicts
from .config_resolvers import DictResolver


class MlopusConfigLoader(OmegaConfigLoader):
    """Patch of OmegaConfigLoader.

    Enabling the patch
    ==================

        .. code-block:: python

            # <your_package>/settings.py
            from mlopus.kedro import MlopusConfigLoader

            CONFIG_LOADER_CLASS = MlopusConfigLoader

    `dict`-like `get` behavior
    ==========================
    Assuming there are no files matching ``conf/<env>/credentials*``, the
    following example would raise an exception with the original OmegaConfigLoader:

        .. code-block:: python

            config.get("credentials")  # returns: None
            config.get("credentials", {})  # returns: {}

    Any scope overrides (CLI)
    =========================

        .. code-block:: bash

            kedro run ... --params globals.key=value
            kedro run ... --params catalog.key=value
            kedro run ... --params parameters.key=value
            kedro run ... --params credentials.key=value

    Any scope overrides (Python)
    ============================

        .. code-block:: python

            with KedroSession.create(
                ...,
                extra_params={
                    "globals": {"key": "value"},
                    "catalog": {"key": "value"},
                    "parameters": {"key": "value"},
                    "credentials": {"key": "value"},
                },
            ) as session:
                ...
    """

    def __init__(
        self,
        conf_source: str,
        env: str | None = None,
        runtime_params: dict[str, Any] | None = None,
        *,
        config_patterns: dict[str, list[str]] | None = None,
        base_env: str = "base",
        default_run_env: str | None = None,
        custom_resolvers: dict[str, typing.Callable] | None = None,
        merge_strategy: dict[str, str] | None = None,
    ):
        (config_patterns := config_patterns or {}).setdefault("globals", ["globals*", "globals*/**", "**/globals*"])

        super().__init__(
            conf_source,
            env or default_run_env or "local",
            runtime_params,
            config_patterns=config_patterns,
            base_env=base_env,
            default_run_env=default_run_env,
            custom_resolvers=custom_resolvers,
            merge_strategy=merge_strategy,
        )

        # Register extra scopes as config resolvers
        for scope in self.config_patterns:
            if scope not in ("globals", "catalog", "parameters", "credentials"):
                DictResolver(self, prefix=[scope]).register(scope)

        # Enforce that runtime params (overrides) specify one of the available config scopes
        for scope in self.runtime_params:
            if scope not in self.config_patterns:
                raise ValueError(
                    f"Unrecognized scope '{scope}' in runtime params (overrides), "
                    f"expected one of: {list(self.config_patterns.keys())} "
                )

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like `get` returning either `None` or the specified `default`."""
        try:
            return super().get(key, default)
        except MissingConfigException:
            return default

    @typing.no_type_check
    def load_and_merge_dir_config(  # noqa: PLR0913
        self,
        conf_path: str,
        patterns: Iterable[str],
        key: str,
        processed_files: set,
        read_environment_variables: bool | None = False,
    ) -> dict[str, Any]:
        """Omit `key` in call to `super` to avoid an early apply of runtime params (overrides)."""
        return super().load_and_merge_dir_config(conf_path, patterns, "", processed_files, read_environment_variables)

    def __getitem__(self, key: str) -> dict[str, Any]:
        """Apply runtime params (overrides) after the config of all matching files has been merged."""
        return dicts.deep_merge(
            super().__getitem__(key),
            self.runtime_params.get(key, {}),
        )
