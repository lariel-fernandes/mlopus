from typing import Any, Sequence, Mapping, Tuple, Hashable, Dict, TypeVar, Iterable

T = TypeVar("T")

AnyDict = Dict[str, Any]


class _Missing:
    pass


_MISSING = _Missing()


def set_reserved_key(_dict: Dict[T, Any] | None, key: T, val: Any) -> Dict[T, Any]:
    """Set key in dict but raise exception if it was already present."""
    if key in (_dict := {} if _dict is None else _dict):
        raise KeyError(f"Reserved key: {key}")
    _dict[key] = val
    return _dict


def map_leaf_vals(data: dict, mapper: callable) -> dict:
    """Recursively map the leaf-values of a dict."""
    new = {}
    for key, val in data.items():
        if isinstance(val, dict):
            mapped = map_leaf_vals(val, mapper)
        elif isinstance(val, (tuple, list, set)):
            mapped = type(val)(mapper(x) for x in val)
        else:
            mapped = mapper(val)
        new[key] = mapped

    return new


def set_nested(_dict: Dict[Hashable, Any], keys: Sequence[Hashable], value: Any) -> Dict[Hashable, Any]:
    """Given keys [a, b, c], set _dict[a][b][c] = value"""
    target = _dict

    for key in keys[:-1]:
        if key not in target:
            target[key] = {}
        target = target[key]

    target[keys[-1]] = value
    return _dict


def flatten(_dict: Mapping) -> Dict[Tuple[str, ...], Any]:
    """Flatten dict turning nested keys into tuples."""

    def _flatten(__dict: Mapping, prefix: Tuple[Hashable, ...]) -> dict:
        flat = {}

        for key, val in __dict.items():
            key = (*prefix, key)

            if isinstance(val, Mapping):
                flat.update(_flatten(val, prefix=key))
            else:
                flat[key] = val

        return flat

    return _flatten(_dict, prefix=())


def unflatten(_dict: Iterable[Tuple[Tuple[str, ...], Any]] | Mapping[Tuple[str, ...], Any]) -> Dict[str, Any]:
    """Turn dict with top-level tuple keys into nested keys."""
    result = {}

    for key, val in _dict.items() if isinstance(_dict, Mapping) else _dict:
        if isinstance(key, tuple):
            set_nested(result, key, val)

    return result
