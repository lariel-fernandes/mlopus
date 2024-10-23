import importlib
import inspect
import re
from pathlib import Path
from typing import Type, Any, Set, Literal, Dict, Tuple

import importlib_metadata
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from mlopus.utils import urls

Dist = importlib_metadata.Distribution

VersionConstraint = Literal["==", "~=", "^", ">="]


class Patterns:
    """Patterns used in packaging inspection."""

    EXTRA_REQ = re.compile(r'^(?P<pkg>[\w.-]+)(?P<specifier>.*); extra == "(?P<extra>\w+)"$')


def get_dist(name: str, strict: bool = True) -> Dist | None:
    """Get distribution metadata by name."""
    try:
        return importlib_metadata.distribution(name)
    except importlib_metadata.PackageNotFoundError:
        if strict:
            raise
        return None


def is_editable_dist(dist: Dist) -> bool:
    """Tell if distribution is installed from editable source code."""
    return (origin := dist.origin) and (dir_info := getattr(origin, "dir_info", None)) and dir_info.editable  # noqa


def get_available_dist_extras(dist: Dist) -> Dict[str, Tuple[str, str]]:
    """Get mapping of optional extras that can be installed for the given package distribution.

    Output format: {extra: [(pkg, specifier), ...]}
    """
    extras = {}

    for req in dist.requires:
        if match := Patterns.EXTRA_REQ.fullmatch(req):
            spec = match.group("pkg"), match.group("specifier")
            extras.setdefault(match.group("extra"), []).append(spec)

    return extras


def get_installed_dist_extras(dist: Dist) -> Set[str]:
    """Get list of optional extras currently installed for the given package distribution."""
    installed = set()

    for extra, reqs in get_available_dist_extras(dist).items():
        for pkg, specifier in reqs:
            if not (dist := get_dist(pkg, strict=False)) or not check_dist(dist, specifier):
                break
        else:
            installed.add(extra)

    return installed


def check_dist(dist: Dist, specifier: str) -> bool:
    """Check if version of package distribution satisfies the specified version constraint."""
    return check_version(dist.version, specifier)


def check_version(actual_version: str, specifier: str) -> bool:
    """Check if version satisfies constraint."""
    return Version(actual_version) in SpecifierSet(
        ",".join(_convert_caret_specifier(x) if x.startswith("^") else x for x in specifier.split(","))
    )


def pkg_dist_of_cls(cls: Type[Any]) -> Dist:
    """Find the package distribution of a class based on its top module location."""
    top_module = importlib.import_module(cls.__module__.split(".", 1)[0])
    init_file = Path(inspect.getfile(top_module))  # __init__.py file in top module

    for dist in importlib_metadata.distributions():
        for file in dist.files:
            if dist.locate_file(file) == init_file:
                return dist

    for dist in importlib_metadata.distributions():
        if (
            (origin := dist.origin)
            and (url := urls.parse_url(origin.url)).scheme == "file"
            and init_file.is_relative_to(url.path)  # noqa
        ):
            return dist

    raise RuntimeError(f"Distribution not found for {cls}")


def _convert_caret_specifier(caret_spec: str) -> str:
    """Convert a caret version specifier (e.g.: ^1.2.3) to a specifier supported by SpecifierSet."""
    version = Version(caret_spec.removeprefix("^"))

    if version.major > 0:
        upper = f"{version.major + 1}"
    elif version.minor > 0:
        upper = f"0.{version.minor + 1}"
    else:
        upper = f"0.0.{version.micro + 1}"

    return f">={version},<{upper}"
