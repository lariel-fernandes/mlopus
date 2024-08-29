import numpy as np
import pytest

from mlopus_kedro_example.model import AnnModel, AnnParams


@pytest.fixture
def vector_size() -> int:
    return 10


@pytest.fixture
def ann_params(vector_size: int) -> AnnParams:
    return AnnParams(
        trees=3,
        metric="angular",
        vector_size=vector_size,
    )


@pytest.fixture(scope="session")
def seed() -> int:
    return 42


@pytest.fixture(scope="session")
def num_vectors() -> int:
    return 20


@pytest.fixture
def vectors(seed: int, num_vectors: int, vector_size: int) -> np.ndarray:
    np.random.seed(seed)
    return np.random.randn(num_vectors, vector_size)


@pytest.fixture
def ann_model(num_vectors: int, ann_params: AnnParams, vectors: np.ndarray) -> AnnModel:
    return AnnModel(
        data=vectors,
        params=ann_params,
        labels=np.array([str(i) for i in range(num_vectors)]),
    )
