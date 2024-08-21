from abc import abstractmethod
from typing import Mapping

from mlopus.utils import pydantic
from . import contract
from .common import schema


class EntityApi(schema.BaseEntity):
    """Self-updating entity with API handle."""

    api: contract.MlflowApiContract = pydantic.Field(default=None, exclude=True, repr=False)

    def using(self, api: contract.MlflowApiContract) -> "EntityApi":
        """Switch API handle."""
        self.api = api
        return self

    def update(self) -> "EntityApi":
        """Use API to get latest data for this entity and update this entity in place."""
        return self._use_values_from(self._get_latest_data())

    def _use_values_from(self, other: schema.BaseEntity) -> "EntityApi":
        """Update this entity in place with values from the other entity. Return a reference to this entity."""
        _ = [setattr(self, k, v) for k, v in other.items()]
        return self

    @abstractmethod
    def _get_latest_data(self) -> schema.BaseEntity:
        """Use API to get latest data for this entity."""

    @abstractmethod
    def set_tags(self, tags: Mapping) -> "EntityApi":
        """Set tags on this entity."""
