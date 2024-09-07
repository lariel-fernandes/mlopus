import typing
from pathlib import Path
from typing import Iterator, Mapping

from mlopus.utils import dicts, pydantic, mongo, urls
from . import entity, contract
from .common import schema, decorators
from .run import RunApi

RunIdentifier = contract.RunIdentifier


class ExpApi(schema.Experiment, entity.EntityApi):
    """Experiment metadata with MLflow API handle."""

    def _get_latest_data(self) -> schema.Experiment:
        """Get latest data for this entity. Used for self update after methods with the `require_update` decorator."""
        return self.api.get_exp(self)

    @property
    def url(self) -> str:
        """This experiment's URL."""
        return self.api.get_exp_url(self)

    @pydantic.validate_arguments
    def find_runs(self, query: mongo.Query | None = None, sorting: mongo.Sorting | None = None) -> Iterator[RunApi]:
        """Search runs belonging to this experiment with query in MongoDB query language.

        :param query: Query in MongoDB query language.
        :param sorting: Sorting criteria (e.g.: `[("asc_field", 1), ("desc_field", -1)]`).
        """
        results = self.api.find_runs(dicts.set_reserved_key(query, key="exp.id", val=self.id), sorting)
        return typing.cast(Iterator[RunApi], results)

    def cache_meta(self) -> "ExpApi":
        """Fetch latest metadata for this experiment and save it to cache."""
        return self._use_values_from(self.api.cache_exp_meta(self))

    def export_meta(self, target: Path) -> "ExpApi":
        """Export experiment metadata cache to target.

        :param target: Cache export path.
        """
        return self._use_values_from(self.api.export_exp_meta(self, target))

    @pydantic.validate_arguments
    def create_run(
        self,
        name: str | None = None,
        tags: Mapping | None = None,
        repo: str | urls.Url | None = None,
        parent: RunIdentifier | None = None,
    ) -> RunApi:
        """Declare a new run in this experiment to be used later.

        :param name: See :attr:`schema.Run.name`.
        :param tags: See :attr:`schema.Run.tags`.
        :param repo: See :paramref:`~mlopus.mlflow.BaseMlflowApi.create_run.repo`.
        :param parent: Parent run ID or object.
        """
        return typing.cast(RunApi, self.api.create_run(self, name, tags, repo, parent))

    @pydantic.validate_arguments
    def start_run(
        self,
        name: str | None = None,
        tags: Mapping | None = None,
        repo: str | urls.Url | None = None,
        parent: RunIdentifier | None = None,
    ) -> RunApi:
        """Start a new run in this experiment.

        :param name: See :attr:`schema.Run.name`.
        :param tags: See :attr:`schema.Run.tags`.
        :param repo: See :paramref:`~mlopus.mlflow.BaseMlflowApi.start_run.repo`.
        :param parent: Parent run ID or object.
        """
        return typing.cast(RunApi, self.api.start_run(self, name, tags, repo, parent))

    @decorators.require_update
    def set_tags(self, tags: Mapping) -> "ExpApi":
        """Set tags on this experiment.

        :param tags: See :attr:`schema.Experiment.tags`.
        """
        self.api.set_tags_on_exp(self, tags)
        return self
