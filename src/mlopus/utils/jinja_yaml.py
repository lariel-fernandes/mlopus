import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, Iterator, Tuple, Callable
import logging
import jinja2
import yaml

from .dicts import deep_merge
from .import_utils import fq_name

logger = logging.getLogger(__name__)

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
    file_extensions: set[str] | None = None,
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
    :param file_extensions: File extensions to use. Defaults to `{'.yml', '.yaml'}`. The leading dot is ignored.
    """

    namespace_files: dict[str, list[Path]] = defaultdict(list)
    for namespace, file_path in _iter_files_with_namespaces(Path(base_path), file_extensions or {".yml", ".yaml"}):
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
        result[namespace] = {}

        for file in namespace_files[namespace]:
            logger.debug(
                "Loading config file. %s",
                details := {"namespace": namespace, "filepath": str(file.relative_to(base_path))},
            )
            rendered = jinja_env.from_string(file.read_text()).render(**context)
            parsed = yaml.safe_load(rendered or "{}")
            assert isinstance(parsed, dict), "Config file parsing must result in a dict. %s" % {
                **details,
                "actual": fq_name(type(parsed)),
            }
            result[namespace] |= parsed

        if ns_overrides := (overrides or {}).get(namespace):
            result[namespace] = deep_merge(result[namespace], ns_overrides)  # noqa

        context[namespace] = result[namespace]

    return result


def _iter_files_with_namespaces(base_path: Path, extensions: list[str]) -> Iterator[Tuple[str, Path]]:
    """Iterate (namespace, path) tuples for every YAML file in the specified path.

    Namespaces are determined by the file's top dir or by its name before any double underscores.
    Examples: <namespace>/subdir/file.yaml, <namespace>__suffix.yml, <namespace>.yaml, etc
    """
    for file_path in (
        x
        for x in base_path.rglob("*.*")
        if x.is_file() and x.suffix.removeprefix(".") in [ext.removeprefix(".") for ext in extensions]
    ):
        rel_path = file_path.relative_to(base_path)
        namespace = rel_path.parts[0] if len(rel_path.parts) > 1 else file_path.stem.split("__")[0]
        yield namespace, file_path
