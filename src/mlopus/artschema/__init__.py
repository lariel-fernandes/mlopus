"""This module provides a framework for defining
reusable dumpers/loaders for custom model/dataset classes.

Schemas can be used standalone or registered in the tags of MLflow entities
such as experiments, runs, models and model versions. In the second case,
artifacts associated to those entities can be saved/loaded/verified
just by providing the alias of a registered schema.
"""

from .catalog import ArtifactsCatalog
from .framework import Dumper, Loader, Schema
from .helpers import load_artifact, get_schema, log_model_version, log_run_artifact, get_schemas
from .specs import (
    parse_load_specs,
    LoadArtifactSpec,
    parse_logart_specs,
    LogArtifactSpec,
    RunArtifact,
    ModelVersionArtifact,
)

from .tags import Tags, ClassSpec

__all__ = [
    "Tags",
    "ClassSpec",
    "Dumper",
    "Loader",
    "Schema",
    "get_schema",
    "get_schemas",
    "load_artifact",
    "log_run_artifact",
    "log_model_version",
    "ArtifactsCatalog",
    "LoadArtifactSpec",
    "parse_load_specs",
    "LogArtifactSpec",
    "parse_logart_specs",
    "RunArtifact",
    "ModelVersionArtifact",
]
