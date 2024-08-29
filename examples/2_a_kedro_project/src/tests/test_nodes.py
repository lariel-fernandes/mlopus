import numpy as np
import pytest

from mlopus_kedro_example import nodes
from mlopus_kedro_example.model import AnnParams, AnnModel
from tests.utils import compare_models, is_sorted


def test_build_vectors(seed: int, num_vectors: int, vector_size: int, vectors: np.ndarray):
    actual_labels, actual_vectors = nodes.BuildVectors(
        seed=seed,
        N=num_vectors,
        D=vector_size,
    ).__call__()

    assert np.all(actual_vectors == vectors)
    assert len(actual_labels) == len(actual_vectors) == num_vectors


def test_build_model(vectors: np.ndarray, ann_params: AnnParams, ann_model: AnnModel):
    actual = nodes.BuildModel(
        trees=ann_params.trees,
        metric=ann_params.metric,
    ).__call__(
        ann_model.labels,
        vectors,
    )

    compare_models(actual, ann_model)



@pytest.mark.parametrize("k_values", [[4, 8, 16]])
def test_eval_model(k_values: list[int], ann_model: AnnModel):
    actual = nodes.EvalModel(
        k_values=k_values,
    ).__call__(
        ann_model,
    )

    for agg_mode, dist_by_k in actual.items():
        print(agg_mode)  # aggregation mode (e.g: avg, max, min)
        assert is_sorted(dist_by_k.values(), ascending=True)  # avg/max/min neighbour distances grow with K value
