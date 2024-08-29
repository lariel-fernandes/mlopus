import numpy as np

from mlopus_kedro_example.model import AnnModel


def is_sorted(a: np.ndarray, ascending: bool) -> bool:
    if not isinstance(a, np.ndarray):
        if hasattr(a, "__iter__"):
            a = np.array(list(a))
        else:
            raise TypeError(a)

    left = a[..., :-1]
    right = a[..., 1:]
    return np.all(left < right) if ascending else np.all(left > right)


def compare_models(a: AnnModel, b: AnnModel):
    assert a.params == b.params
    assert np.all(a.labels == b.labels)
    assert np.all(a.get_vectors_by_labels(a.labels) == b.get_vectors_by_labels(b.labels))
    assert all(
        np.all(_actual == _expected) for _actual, _expected in
        zip(
            a.get_ann_by_labels(a.labels, len(a.labels) - 1),
            b.get_ann_by_labels(b.labels, len(b.labels) - 1)
        )
    )
