from pathlib import Path
from typing import Any, Literal


NamespacedConfigs = dict[str, dict[str, Any]]


def load_jinja_yaml_configs(
    base_path: str | Path,
    overrides: NamespacedConfigs | None = None,
    namespaces: list[str] | None = None,
    load_mode: Literal["all", "explicit"] | None = None,
    extra_namespaces: NamespacedConfigs | None = None,
    include_env: bool = False,
    env_namespace: str = "env",
) -> NamespacedConfigs:
    """
    Load namespaced configs from jinja-templated YAML files.

    :param base_path: Base path for jinja-templated YAML files.
    :param overrides: After each namespace is loaded, the respective overrides are applied via deep-merge.
    :param extra_namespaces: Extra namespaces made available for interpolation in YAML files.
    :param include_env: Whether to include an extra namespace that exposes environment variables.
    :param env_namespace: The name of the extra namespace for environment variables.
    :param namespaces: Namespaces to load, in order of precedence.
    :param load_mode:
        `explicit`: Only load the specified namespaces. This is the default if at least one namespace is specified.
        `all`: Load the specified namespaces first, then the remaining in alphanumerical order. This is the default if no namespaces are specified.
    """

    # TODO: iterate all .yml or .yaml files in `base_path` recursively assigning them to namespaces.
    #       If the file is inside a dir, the namespace is the dir. Otherwise, it's the file name before a double underscore (e.g. my_namespace__suffix.yml)

    # TODO: determine the default of the load mode if necessary. Iterate the namespaces. For each namespace, concatenate the content of all files in memory then parse them as a jinja template, then load as yaml content, assert the result is a dict
    #       when parsing a namespace, make the previously parsed namespaces available for jinja values injection
    #       after parsing each namespace, if there are overrides for it, use deep_merge to combine it with its respective overrides
