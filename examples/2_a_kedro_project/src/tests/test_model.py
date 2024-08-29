import numpy as np
import pytest

from mlopus_kedro_example.model import AnnModel, AnnParams
from tests.utils import is_sorted


class TestModel:

    @pytest.mark.parametrize("n_labels", [10])
    def test_alloc_vectors(self, n_labels: int, vector_size: int, ann_model: AnnModel):
        array = ann_model.alloc_out_buffer__get_vectors_by_labels(n_labels)
        assert array.shape == (n_labels, vector_size)

    @pytest.mark.parametrize("n_labels", [10])
    @pytest.mark.parametrize("n_neighbours", [3])
    def test_alloc_ann(self, n_labels: int, n_neighbours: int, ann_params: AnnParams, ann_model: AnnModel):
        labels, dist = ann_model.alloc_out_buffers__get_ann_by_labels(n_labels, n_neighbours)
        assert labels.shape == dist.shape == (n_labels, n_neighbours)

    def test_vectors(self, vectors: np.ndarray, ann_model: AnnModel):
        actual = ann_model.get_vectors_by_labels(ann_model.labels)
        assert np.allclose(vectors, actual)

    @pytest.mark.parametrize("n_neighbours", [3])
    def test_ann(self, n_neighbours: int, ann_model: AnnModel):
        labels, dist = ann_model.get_ann_by_labels(ann_model.labels, n_neighbours)
        assert is_sorted(dist, ascending=True)  # distances are sorted from closer to further
        assert np.all(labels[:, 0] != ann_model.labels)  # first neighbour for each label cannot be the same label
