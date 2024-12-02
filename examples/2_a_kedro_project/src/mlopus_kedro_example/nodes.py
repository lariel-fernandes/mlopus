"""Kedro Node functions."""
from typing import Tuple

import numpy as np
from mlopus.kedro import node_tools

from .model import AnnModel, AnnParams


class BuildVectors(node_tools.NodeFunc):
    """Build random vectors."""
    N: int
    D: int
    seed: int

    def __call__(self) -> Tuple[np.ndarray, np.ndarray]:
        np.random.seed(self.seed)
        labels = [f"L{i}" for i in range(self.N)]
        vectors = np.random.randn(self.N, self.D)
        return np.array(labels), np.array(vectors)


class BuildModel(node_tools.NodeFunc):
    """Build vector index model."""
    trees: int
    metric: str

    def __call__(self, labels: np.ndarray, vectors: np.ndarray) -> AnnModel:
        return AnnModel(
            data=vectors,
            labels=labels,
            params=AnnParams(
                trees=self.trees,
                metric=self.metric,
                vector_size=vectors.shape[1],
            ),
        )


class EvalModel(node_tools.NodeFunc):
    """Evaluate model in terms of distances at K."""
    k_values: list[int]

    async def __acall__(self, model: AnnModel) -> dict:
        metrics = {}

        _, dist = model.get_ann_by_labels(model.labels, max_neighbours=max(self.k_values))

        for k in sorted(self.k_values):
            metrics.setdefault("max_dist_at", {})[k] = float(dist[:, :k].max())
            metrics.setdefault("avg_dist_at", {})[k] = float(dist[:, :k].mean())

        return metrics
