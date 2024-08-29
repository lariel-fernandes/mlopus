from typing import Any, List, Mapping

from omegaconf import OmegaConf

from mlopus.utils import dicts


class _Missing:
    """Placeholder for missing default."""


_MISSING = _Missing()


class Resolver:
    """Base class for self-registering custom config resolvers."""

    def register(self, name: str, replace: bool = True):
        """Register this resolver."""
        OmegaConf.register_new_resolver(name, self, replace=replace)

    def __call__(self, *_) -> Any:
        """Resolve config value from args."""


class DictResolver(Resolver):
    """Resolver for OmegaConf based on static dict."""

    def __init__(self, conf: Mapping, prefix: List[str] | None = None):
        self._conf = conf
        self._prefix = prefix or []

    def __call__(self, key: str | None = None, default=_MISSING):
        """Resolve nested key from dict data."""
        if not key:
            return dict(self._conf)
        if (val := dicts.get_nested(self._conf, self._prefix + key.split("."), default=default)) is _MISSING:
            raise KeyError(key)
        return val
