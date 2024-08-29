"""AI Model Classes."""
import logging
from pathlib import Path
from typing import Tuple, Literal, Sequence

import annoy
import numpy as np
from pydantic.v1 import BaseModel, validate_arguments

from .exceptions import ParamOutOfBounds

logger = logging.getLogger(__name__)


class AnnParams(BaseModel):
    """Params for `AnnModel`."""
    trees: int
    vector_size: int
    metric: Literal["angular", "euclidean", "manhattan", "hamming", "dot"]


class AnnModel:
    """A simple ANN model based on a static vector index and label lookups."""

    @validate_arguments(config={"arbitrary_types_allowed": True})
    def __init__(self, params: AnnParams, labels: np.ndarray, data: np.ndarray | Path):
        assert labels.ndim == 1

        self.params = params
        self.labels = labels
        self.n_items = len(labels)
        self.max_neighbours = self.n_items - 1
        self.index = annoy.AnnoyIndex(params.vector_size, params.metric)
        self.labels_to_vector_ids = {label: i for i, label in enumerate(list(labels))}

        if isinstance(data, np.ndarray):
            assert data.shape == (self.n_items, params.vector_size)
            [self.index.add_item(i, vector) for i, vector in enumerate(data)]
            self.index.build(params.trees)
        else:
            self.index.load(str(data))
            assert self.index.get_n_items() == self.n_items

    def get_vectors_by_labels(self, labels: Sequence[str]) -> np.ndarray:
        """Get vectors for labels."""
        vectors = self.alloc_out_buffer__get_vectors_by_labels(len(labels))

        for i, label in enumerate(labels):
            if (_id := self.labels_to_vector_ids.get(label)) is not None:
                vectors[i] = self.index.get_item_vector(_id)

        return vectors

    def get_ann_by_labels(
        self,
        labels: Sequence[str],
        max_neighbours: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get ANN labels and distances for each query label."""
        labels_buff, dist_buff = self.alloc_out_buffers__get_ann_by_labels(len(labels), max_neighbours)

        for i, label in enumerate(labels):
            if (_id := self.labels_to_vector_ids.get(label)) is not None:
                # First neighbour is always the same as the queried item,
                # so we ask for `max_neighbours + 1` and skip the first result.
                _ann_ids, _ann_dist = self.index.get_nns_by_item(_id, n=max_neighbours + 1, include_distances=True)
                dist_buff[i, :max_neighbours] = _ann_dist[1:]
                labels_buff[i, :max_neighbours] = self.labels[_ann_ids[1:]]

        return labels_buff, dist_buff

    def alloc_out_buffer__get_vectors_by_labels(self, n_labels: int) -> np.ndarray:
        """Allocate output buffers for `get_vectors_by_labels`."""
        shape = (n_labels, self.params.vector_size)
        vectors = np.full(shape, dtype=np.float32, fill_value=np.nan)
        return vectors

    def alloc_out_buffers__get_ann_by_labels(self, n_labels: int, max_neighbours: int) -> Tuple[np.ndarray, np.ndarray]:
        """Allocate output buffers for `get_ann_by_labels`."""
        ParamOutOfBounds("max_neighbours", 0, self.n_items).maybe_raise(max_neighbours)
        shape = (n_labels, max_neighbours)
        ann_dist = np.full(shape, dtype=np.float16, fill_value=np.nan)
        ann_labels = np.full(shape, dtype=np.object_, fill_value=float("nan"))
        return ann_labels, ann_dist
