from abc import ABC, abstractmethod
from typing import Iterable, Mapping, Any

from kedro.pipeline.node import Node, node

from mlopus.utils import pydantic, dicts, string_utils


class NodeFunc(pydantic.BaseModel, ABC):
    """Base class for parameterized functions to be used in Kedro nodes.

    Example:
        # nodes.py
        class MyFunc(NodeFunc):
            p1: ...
            p2: ...

            def __call__(self, x, y):
                # process inputs `(x, y)` using params `(self.p1, self.p2)` and return something

        # parameters.yml
        nodes:
            my_func: {p1: ..., p2: ...}

        # pipeline.py
        def create_pipeline(config):
            my_func = MyFunc.parse_obj(config["parameters"]["nodes"]["my_func"])
            # Alternatively: my_func = MyFunc.default_parser(config)

            return Pipeline([
                MyFunc.to_node(inputs=[...], outputs=...),
            ])

        # pipeline_registry.py
        def register_pipelines():
            return {
                "my_pipeline": pipeline_factory(create_pipeline),
            }
    """

    @abstractmethod
    def __call__(self, **__):
        """Node function implementation."""

    @classmethod
    def _default_func_name(cls) -> str:
        """Default name of this node function."""
        return string_utils.camel_to_snake(cls.__name__)

    @classmethod
    def _default_config_path(cls) -> str:
        """Default config key-path for obtaining the params that configure this function."""
        return f"parameters.{cls._default_func_name()}"

    @classmethod
    def default_parser(cls, conf: Mapping, **extra_params: Any) -> "NodeFunc":
        """Initialize this function using params obtained from the configuration at the default key-path."""
        params = dicts.get_nested(conf, cls._default_config_path().split("."))
        return cls.parse_obj(dicts.deep_merge(params, extra_params))

    def to_node(
        self,
        inputs: str | list[str] | dict[str, str] | None = None,
        outputs: str | list[str] | dict[str, str] | None = None,
        *,
        name: str | None = None,
        tags: str | Iterable[str] | None = None,
        confirms: str | list[str] | None = None,
        namespace: str | None = None,
    ) -> Node:
        """Wrap this function in a Kedro node."""
        return node(
            func=self,
            inputs=inputs,
            outputs=outputs,
            name=name or self._default_func_name(),
            tags=tags,
            confirms=confirms,
            namespace=namespace,
        )
