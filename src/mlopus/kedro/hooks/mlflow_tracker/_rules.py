import re
from abc import abstractmethod, ABC
from typing import Literal, List, Tuple, Iterable, TypeVar, Generic, Dict, Mapping

from mlopus.utils import pydantic, dicts
from ._pipeline import _Node, _Pipeline

T = TypeVar("T")  # Any type
K = TypeVar("K")  # Any type of key
V = TypeVar("V")  # Any type of val

Effect = Literal["exclude-all", "include-only"]


class _BaseRule(pydantic.BaseModel, ABC, Generic[T]):
    effect: Effect
    keys: List[str] = []
    expr: List[re.Pattern] = []

    def allow(self, subject: T) -> bool:
        match = self.match(subject)
        return (match and self.effect == "include-only") or (not match and self.effect == "exclude-all")

    def match(self, subject: T) -> bool:
        subject = self.process_subject(subject)

        if self.keys and subject in self.keys:
            return True

        if self.expr and any(expr.match(subject) for expr in self.expr):
            return True

        return False

    @abstractmethod
    def process_subject(self, subject: T) -> str:
        pass


class _BaseRuleSet(pydantic.BaseModel, Generic[T, K, V]):
    rules: List[_BaseRule[T]] = []

    def allow(self, subject: T) -> bool:
        return all(rule.allow(subject) for rule in self.rules)

    def apply(self, subjects: Iterable[T]) -> Dict[K, V]:
        return {self.process_key(s): self.process_val(s) for s in subjects if self.allow(s)}

    @abstractmethod
    def process_key(self, subject: T) -> K:
        pass

    @abstractmethod
    def process_val(self, subject: T) -> V:
        pass


class _Rule(_BaseRule[Tuple[str, dict]]):
    def process_subject(self, subject: Tuple[str, dict]) -> str:
        return subject[0]


class _RuleSet(_BaseRuleSet[Tuple[str, dict], str, dict]):
    rules: List[_Rule] = []

    def process_key(self, subject: Tuple[str, dict]) -> str:
        return subject[0]

    def process_val(self, subject: Tuple[str, dict]) -> dict:
        return subject[1]


class _NodeRule(_BaseRule[_Node]):
    keys: List[str] = pydantic.Field(default_factory=list, alias="names")
    tags: List[str] = []

    def process_subject(self, subject: _Node) -> str:
        return subject.name

    def match(self, node: _Node) -> bool:
        if self.tags and set(self.tags).intersection(node.tags):
            return True

        return super().match(node)


class _NodeRuleSet(_BaseRuleSet[_Node, str, dict]):
    rules: List[_NodeRule] = []

    def allow(self, subject: _Node) -> bool:
        return subject.func is not None and super().allow(subject)

    def process_key(self, key: _Node) -> str:
        return key.name

    def process_val(self, subject: _Node) -> dict:
        return subject.func.conf


class _ScopedRuleSet(_RuleSet):
    scopes: List[str] = []

    def apply(self, scoped_subjects: Mapping[str, Dict[str, dict]]) -> Dict[str, Dict[K, V]]:
        return {
            scope: data  # noqa
            for scope in self.scopes
            if (subject := scoped_subjects.get(scope)) and (data := _RuleSet.apply(self, subject.items()))
        }


class _PrefixSuffix(pydantic.BaseModel):
    prefix: List[str] = []
    suffix: List[str] = []

    def apply_prefix(self, data: dict) -> dict:
        return dicts.new_nested(self.prefix, data) if self.prefix else data

    def apply_suffix(self, data: dict) -> dict:
        return dicts.map_leaf_vals(data, lambda val: dicts.new_nested(self.suffix, val)) if self.suffix else data

    def apply(self, data: dict) -> dict:
        return self.apply_prefix(self.apply_suffix(data))


class _PrefixSuffixRuleSet(_PrefixSuffix, _RuleSet):
    def apply(self, subjects: Iterable[Tuple[str, dict]]) -> dict:
        return _PrefixSuffix.apply(self, _RuleSet.apply(self, subjects))


class _ScopedPrefixSuffixRuleSet(_PrefixSuffix, _ScopedRuleSet):
    def apply(self, scoped_subjects: Dict[str, T]) -> Dict[K, V]:
        return _PrefixSuffix.apply(self, _ScopedRuleSet.apply(self, scoped_subjects))


class _PipelinesRuleSet(_PrefixSuffix, _NodeRuleSet):
    prepend_pipeline: bool = True

    def _apply(self, pipelines: Dict[str, _Pipeline]) -> Iterable[Tuple[str, dict]]:
        for pipeline_name, pipeline in pipelines.items():
            node_confs = _NodeRuleSet.apply(self, pipeline.nodes.values())
            node_confs = self.apply_suffix(node_confs)
            if self.prepend_pipeline:
                yield pipeline_name, node_confs
            else:
                yield from node_confs.items()

    def apply(self, pipelines: Dict[str, _Pipeline]) -> dict:
        return self.apply_prefix(dict(self._apply(pipelines)))
