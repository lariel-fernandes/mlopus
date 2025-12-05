import copy
from collections.abc import KeysView
from typing import Any, Callable

from kedro.config import AbstractConfigLoader

from mlopus.utils.jinja_yaml import NamespacedConfigs, load_jinja_yaml_configs, LoadMode, DEFAULT_ENV_NAMESPACE


class JinjaYamlConfigLoader(AbstractConfigLoader):
    """Kedro config loader for Jinja-templated YAML files."""

    def __init__(
        self,
        conf_source: str | None,
        env: str | None = None,
        runtime_params: NamespacedConfigs | None = None,
        *,
        namespace_mappings: dict[str, str] | None = None,
        namespaces: list[str] | None = None,
        load_mode: LoadMode | None = None,
        extra_namespaces: NamespacedConfigs | None = None,
        expose_env: bool = False,
        env_namespace: str = DEFAULT_ENV_NAMESPACE,
        custom_filters: dict[str, Callable] | None = None,
        file_extensions: set[str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            conf_source=(conf_source := conf_source or "conf"),
            env=env,
            runtime_params=runtime_params,
            **kwargs,
        )

        self._namespace_mappings = namespace_mappings or {}

        self._config = load_jinja_yaml_configs(
            base_path=conf_source,
            overrides=runtime_params,
            namespaces=namespaces,
            load_mode=load_mode,
            extra_namespaces=extra_namespaces,
            expose_env=expose_env,
            env_namespace=env_namespace,
            custom_filters=custom_filters,
            file_extensions=file_extensions,
        )

        # The following config namespaces are required by Kedro, so we give them a default if necessary.
        for required_key in ["globals", "catalog", "parameters", "credentials"]:
            if required_key not in self._namespace_mappings:
                self._config.setdefault(required_key, {})

    def __getitem__(self, item: str) -> dict[str, Any]:
        return copy.deepcopy(self._config[self._namespace_mappings.get(item) or item])

    def __setitem__(self, key: str, value: Any) -> None:
        raise RuntimeError(f"{self.__class__.__name__} is read-only")

    def keys(self) -> KeysView:
        return KeysView(list(self._config.keys()) + list(self._namespace_mappings.keys()))  # noqa

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({list(self.keys())})"
