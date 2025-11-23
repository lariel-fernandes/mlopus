import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, Iterator, Tuple, Callable

import jinja2
import yaml

from .dicts import deep_merge

NamespacedConfigs = dict[str, dict[str, Any]]

LoadMode = Literal["explicit", "all"]

DEFAULT_ENV_NAMESPACE = "env"


def load_jinja_yaml_configs(
    base_path: str | Path,
    overrides: NamespacedConfigs | None = None,
    namespaces: list[str] | None = None,
    load_mode: LoadMode | None = None,
    extra_namespaces: NamespacedConfigs | None = None,
    expose_env: bool = False,
    env_namespace: str = DEFAULT_ENV_NAMESPACE,
    custom_filters: dict[str, Callable] | None = None,
) -> NamespacedConfigs:
    """
    Load namespaced configs from jinja-templated YAML files.

    :param base_path: Base path for jinja-templated YAML files.
    :param overrides: After each namespace is loaded, the respective overrides are applied via deep-merge.
    :param namespaces: Namespaces to load, in order of precedence.
    :param load_mode:
        `explicit`: Only load the specified namespaces. This is the default if at least one namespace is specified.
        `all`: Load the specified namespaces first, then the remaining in alphanumerical order. This is the default if no namespaces are specified.

    :param extra_namespaces: Extra namespaces made available for interpolation in YAML files.
    :param expose_env: Whether to include an extra namespace that exposes environment variables.
    :param env_namespace: The name of the extra namespace for environment variables.

    :param custom_filters: Custom jinja2 filters for manipulating values in the template.
    """

    namespace_files: dict[str, list[Path]] = defaultdict(list)
    for namespace, file_path in _iter_files_with_namespaces(Path(base_path)):
        namespace_files[namespace].append(file_path)

    for files in namespace_files.values():
        files.sort()

    if load_mode is None:
        load_mode = "explicit" if namespaces else "all"

    namespaces_to_load = namespaces or []
    if load_mode == "all":
        namespaces_to_load += [ns for ns in sorted(namespace_files.keys()) if ns not in namespaces]

    context = {
        **(extra_namespaces or {}),
        **({env_namespace: dict(os.environ)} if expose_env else {}),
    }
    result: NamespacedConfigs = {}

    jinja_env = jinja2.Environment()
    jinja_env.filters.update(custom_filters or {})

    for namespace in namespaces_to_load:
        raw = "\n".join(file.read_text() for file in namespace_files[namespace])
        rendered = jinja_env.from_string(raw).render(**context)
        parsed = yaml.safe_load(rendered or "{}")
        assert isinstance(parsed, dict), f"Namespace '{namespace}' must be a dict, got {type(parsed).__name__}"
        result[namespace] = context[namespace] = (
            deep_merge(parsed, overrides[namespace]) if overrides and namespace in overrides else parsed
        )

    return result


def _iter_files_with_namespaces(base_path: Path) -> Iterator[Tuple[str, Path]]:
    """Iterate (namespace, path) tuples for every YAML file in the specified path.

    Namespaces are determined by the file's top dir or by its name before any double underscores.
    Examples: <namespace>/subdir/file.yaml, <namespace>__suffix.yml, <namespace>.yaml, etc
    """
    for file_path in (x for x in base_path.rglob("*.y*ml") if x.suffix in (".yml", ".yaml")):
        rel_path = file_path.relative_to(base_path)
        namespace = rel_path.parts[0] if len(rel_path.parts) > 1 else file_path.stem.split("__")[0]
        yield namespace, file_path
