import sys
from abc import ABC
from typing import Generic, TypeVar

import pydantic
import pytest

import mlopus


class Artifact:
    pass


A = TypeVar("A", bound=Artifact)


class Dumper(mlopus.artschema.Dumper[A], ABC, Generic[A]):
    pass


D = TypeVar("D", bound=Dumper)


class Loader(mlopus.artschema.Loader[A, D], ABC, Generic[A, D]):
    pass


L = TypeVar("L", bound=Loader)


class Schema(mlopus.artschema.Schema[A, D, L], ABC, Generic[A, D, L]):
    pass


class Artifact2(Artifact):
    pass


A2 = TypeVar("A2", bound=Artifact2)


class Dumper2(Dumper[A2], ABC, Generic[A2]):
    pass


D2 = TypeVar("D2", bound=Dumper2)


class Loader2(Loader[A2, D2], ABC, Generic[A2, D2]):
    pass


L2 = TypeVar("L2", bound=Loader2)


class Schema2(Schema[A2, D2, L2], ABC, Generic[A2, D2, L2]):
    pass


def test_type_param_inference():
    assert Dumper._get_artifact_type() == Artifact
    assert Loader._get_artifact_type() == Artifact
    assert Schema._get_artifact_type() == Artifact
    assert Loader._get_dumper_type() == Dumper
    assert Schema._get_dumper_type() == Dumper
    assert Schema._get_loader_type() == Loader

    assert Dumper2._get_artifact_type() == Artifact2
    assert Loader2._get_artifact_type() == Artifact2
    assert Schema2._get_artifact_type() == Artifact2
    assert Loader2._get_dumper_type() == Dumper2
    assert Schema2._get_dumper_type() == Dumper2
    assert Schema2._get_loader_type() == Loader2


sys.path.append("examples/1_introduction/code/my-schemas")

from my_schemas import foobar  # noqa


def test_schema(temp_dir):
    data = {"some_data": {"foo": "bar"}}
    artifact = foobar.Artifact.model_validate(data)
    schema: mlopus.artschema.Schema = foobar.Schema()
    custom_loader: mlopus.artschema.Loader = foobar.Loader(max_files=0)

    with temp_dir() as tmp:
        stg = tmp / "stg"  # temporary staging dir

        # Pass the expected artifact type
        schema.get_dumper(artifact)(stg)

        # Pass a dict that can be parsed into the artifact type
        schema.get_dumper(data)(stg, overwrite=True)

        # Pass a dict that cannot be parsed into the artifact type
        with pytest.raises(pydantic.ValidationError):
            schema.get_dumper({"x": 1})(stg)

        # Try to overwrite without setting the overwrite flag
        with pytest.raises(IsADirectoryError):
            schema.get_dumper(artifact)(stg)

        # Pass a path with a pre-serialized artifact
        assert schema.get_dumper(stg) == stg

        # Load in dry-run mode (just verify)
        assert schema.get_loader(dry_run=True)(stg) == stg

        # Load
        assert schema.get_loader()(stg) == artifact

        # Load with custom loader (max_files=0)
        assert schema.get_loader(custom_loader)(stg).some_data == {}

        # Load with custom loader conf (max_files=0)
        assert schema.get_loader(**custom_loader.dict())(stg).some_data == {}

        class _DictLoader(foobar.Loader):
            def _load(self, path, dumper) -> dict:
                return super()._load(path, dumper).dict()  # noqa

        # Load with implicit conversion (loaded dict to expected pydantic artifact type)
        assert schema.get_loader(_DictLoader())(stg) == artifact
