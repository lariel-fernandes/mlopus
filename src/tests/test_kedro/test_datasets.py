import os
from pathlib import Path

import pytest

import mlopus
from mlopus.kedro.datasets import ArtifactSchemaDataset
from mlopus.utils import import_utils

pytest_plugins = "tests.test_kedro.fixtures"


class Dumper(mlopus.artschema.Dumper[dict]):
    def _dump(self, path: Path, artifact: dict) -> None:
        path.mkdir()
        for key, val in artifact.items():
            (path / f"{key}.txt").write_text(val)

    def _verify(self, path: Path) -> None:
        assert path.is_dir()
        assert all((path / file).is_file() for file in os.listdir(path))


class Loader(mlopus.artschema.Loader[dict, Dumper]):
    def _load(self, path: Path, dumper: Dumper) -> dict:
        return {
            file.removesuffix(".txt"): (path / file).read_text() for file in os.listdir(path) if file.endswith(".txt")
        }


class Schema(mlopus.artschema.Schema[dict, Dumper, Loader]):
    pass


schemas = [Schema, import_utils.fq_name(Schema)]

subjects = [
    {"exp_name": "exp"},
    {"model_name": "model"},
    None,  # with run manager present, defaults to session run
]


class TestArtifactSchemaDataset:
    @pytest.mark.parametrize("schema", list(range(len(schemas))))
    def test_explicit_schema(self, temp_dir, temp_mlflow, schema):
        self._test_artifact_schema_dataset(temp_dir, schema=schemas[schema])

    @pytest.mark.parametrize("subject", list(range(len(subjects))))
    def test_inferred_schema(
        self, temp_dir, temp_kedro_conf, temp_run_manager: mlopus.mlflow.traits.MlflowRunManager, subject
    ):
        tags = mlopus.artschema.Tags().using(Schema, from_package="mlopus")

        temp_run_manager.mlflow_api.get_or_create_exp("exp").set_tags(tags)

        temp_run_manager.mlflow_api.get_or_create_model("model").set_tags(tags)

        temp_run_manager.run.set_tags(tags)  # register artifact schema on session run

        self._test_artifact_schema_dataset(
            temp_dir,
            schema="default",
            subject=subjects[subject],
            mlflow=temp_kedro_conf["globals"]["mlflow"],
        )

    @classmethod
    def _test_artifact_schema_dataset(cls, temp_dir, **kwargs):
        artifact = {str(i): i * "x" for i in range(1, 4)}

        with temp_dir() as tmp:
            dataset = ArtifactSchemaDataset(**kwargs, path=tmp / "path")
            dataset._save(artifact)
            assert artifact == dataset._load()
