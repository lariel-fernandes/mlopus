from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from mlopus_kedro_example.artschema import AnnModelSchema
from mlopus_kedro_example.model import AnnModel
from tests.utils import compare_models

schema = AnnModelSchema()


def test_artschema(vectors: np.ndarray, ann_model: AnnModel) -> None:
    loader = schema.get_loader()
    dumper = schema.get_dumper(ann_model)

    with TemporaryDirectory() as tmp:
        dumper(stg := Path(tmp) / "stg")
        actual = loader(stg)

    compare_models(actual, ann_model)
