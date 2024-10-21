from copy import deepcopy
from typing import Any, Sequence, Mapping, Tuple, Hashable, Dict, TypeVar, List, Iterable

T = TypeVar("T")

AnyDict = Dict[str, Any]


class _Missing:
    pass


_MISSING = _Missing()


def set_if_empty(_dict: dict, key: str, val: Any) -> dict:
    """Set key to val in dict if current val is absent, None or an empty container."""
    if not (current := _dict.get(key)) and current is not False:
        _dict[key] = val
    return _dict


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


def get_nested(_dict: Mapping, keys: Sequence[Hashable], default: Any = _MISSING) -> Any:
    """Given keys [a, b, c], return _dict[a][b][c]"""
    target = _dict

    for idx, key in enumerate(keys):
        try:
            target = target[key]
        except KeyError:
            if default is _MISSING:
                raise KeyError(keys[0 : idx + 1])  # noqa
            return default

    return target


def set_nested(_dict: Dict[Hashable, Any], keys: Sequence[Hashable], value: Any) -> Dict[Hashable, Any]:
    """Given keys [a, b, c], set _dict[a][b][c] = value"""
    target = _dict

    for key in keys[:-1]:
        if key not in target:
            target[key] = {}
        target = target[key]

    target[keys[-1]] = value
    return _dict


def has_nested(_dict: Dict[Hashable, Any], keys: Sequence[Hashable]) -> bool:
    """Given keys [a, b, c], tell if `_dict[a][b][c]` exists."""
    try:
        get_nested(_dict, keys)
        return True
    except KeyError:
        return False


def new_nested(keys: Sequence[Hashable], value: Any) -> Dict[Hashable, Any]:
    """Given keys [a, b, c], produce {a: {b: {c: value}}}"""
    return set_nested({}, keys, value)


def filter_empty_leaves(dict_: Mapping) -> dict:
    """Filter out leaf-values that are None or empty iterables."""
    return unflatten(((k, v) for k, v in flatten(dict_).items() if v or v is False))


def deep_merge(*dicts: dict):
    """Merge dicts at the level of leaf-values."""
    retval = {}

    def _update(tgt: dict, src: Mapping, prefix_keys: List[str]):
        for key, val in src.items():
            _key = prefix_keys + [key]

            if isinstance(val, Mapping) and (val or isinstance(get_nested(tgt, _key, None), Mapping)):
                # Treat value as nested if it's a non-empty dict or if the target is already nested
                _update(tgt, val, _key)
            else:
                # Treat value as a leaf (scalar) otherwise
                set_nested(tgt, _key, deepcopy(val))

    for _dict in dicts:
        _update(retval, _dict, prefix_keys=[])

    return retval


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
