from typing import Dict, Set, Optional

from kedro.pipeline.node import Node

from mlopus.utils import pydantic, import_utils


class _Func(pydantic.BaseModel):
    type: str | None = None
    conf: dict | None = None

    @classmethod
    def from_callable(cls, func_: callable) -> Optional["_Func"]:
        if func := pydantic.as_model_obj(func_):
            return cls(
                type=import_utils.fq_name(type(func)),
                conf=func.dict(),
            )
        return None


class _Node(pydantic.BaseModel, pydantic.ExcludeEmptyMixin):
    name: str = pydantic.Field(exclude=True)
    tags: Set[str] = set()
    inputs: Set[str] = set()
    outputs: Set[str] = set()
    func: _Func | None = None

    @classmethod
    def from_kedro(cls, node: Node) -> "_Node":
        return cls(
            name=node.name,
            tags=node.tags,
            inputs=set(node.inputs),
            outputs=set(node.outputs),
            func=_Func.from_callable(node.func),
        )


class _Pipeline(pydantic.BaseModel, pydantic.ExcludeEmptyMixin):
    namespace: str | None = None
    tags: Set[str] | None = set()
    success: bool | None = None
    error: Exception | None = None
    nodes: Dict[str, _Node] = {}

    def set_error(self, error: Exception | None):
        self.error = error
        self.success = error is None
