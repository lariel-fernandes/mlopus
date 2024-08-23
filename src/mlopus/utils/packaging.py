import importlib
import inspect
import re
from pathlib import Path
from typing import Type, Any, Set, Literal

import importlib_metadata
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from mlopus.utils import urls

Dist = importlib_metadata.Distribution

VersionConstraint = Literal["==", "~", "^", ">="]


class Patterns:
    """Patterns used in packaging inspection."""

    EXTRA_REQ = re.compile(r'^\w+ \(.*\) ; extra == "(?P<extra>\w+)"$')  # extracts optional extra from package req


def get_dist(name: str) -> Dist:
    """Get distribution metadata by name."""
    return importlib_metadata.distribution(name)


def is_editable_dist(dist: Dist) -> bool:
    """Tell if distribution is installed from editable source code."""
    return (origin := dist.origin) and (dir_info := getattr(origin, "dir_info", None)) and dir_info.editable  # noqa


def get_available_dist_extras(dist: Dist) -> Set[str]:
    """Get list of optional extras that can be installed for the given package distribution."""
    return set(dist.metadata.get_all("Provides-Extra"))


def get_installed_dist_extras(dist: Dist) -> Set[str]:
    """Get list of optional extras currently installed for the given package distribution."""
    return {match.group("extra") for x in dist.requires if (match := Patterns.EXTRA_REQ.fullmatch(x))}


def check_dist(dist: Dist, version: str, constraint: VersionConstraint) -> bool:
    """Check if version of package distribution satisfies the specified version constraint."""
    return check_version(dist.version, version, constraint)


def check_version(actual_version: str, required_version: str, constraint: VersionConstraint) -> bool:
    """Check if version satisfies constraint."""
    return Version(actual_version) in SpecifierSet(_convert_specifier(constraint + required_version))


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


def _convert_specifier(version_constraint: str):
    """Convert version constraint to pattern accepted by `packaging.SpecifierSet`"""
    if version_constraint.startswith("^"):
        base_version = version_constraint[1:]
        major, minor, _ = base_version.split(".")
        return f">={base_version},<{int(major)+1}.0.0"
    elif version_constraint.startswith("~"):
        base_version = version_constraint[1:]
        major, minor, _ = base_version.split(".")
        return f">={base_version},<{major}.{int(minor)+1}.0"
    else:
        return version_constraint
