import re
from datetime import datetime
from typing import Dict, Any, TypeVar, Literal, Tuple, Iterable, Callable, List

from mlopus.utils import pydantic, dicts, time_utils

T = TypeVar("T")

Query = Dict[str, Any]

Direction = Literal[1, -1]

Sorting = List[Tuple[str, Direction]]


def preprocess_query_or_doc(obj: dicts.AnyDict) -> dicts.AnyDict:
    """Use safe repr for datetime because mongomock doesn't handle datetime quite well."""
    return dicts.map_leaf_vals(obj, lambda x: time_utils.safe_repr(x) if isinstance(x, datetime) else x)


def deprocess_query_or_doc(obj: dicts.AnyDict) -> dicts.AnyDict:
    """Inverse function of `preprocess_query_or_doc`."""
    return dicts.map_leaf_vals(obj, time_utils.maybe_parse_safe_repr)


def find_all(
    objs: Iterable[T],
    query: Query,
    sorting: Sorting | None = None,
    to_doc: Callable[[T], dicts.AnyDict] = lambda x: x,
    from_doc: Callable[[dicts.AnyDict], T] = lambda x: x,
) -> Iterable[T]:
    """Find all objects matching query in MongoDB query language."""
    from mongomock.mongo_client import MongoClient

    coll = MongoClient(tz_aware=True).db.collection
    coll.insert_many((preprocess_query_or_doc(to_doc(x)) for x in objs))
    docs = coll.find(preprocess_query_or_doc(query), sort=sorting)
    return (deprocess_query_or_doc(from_doc(x)) for x in docs)


class Mongo2Sql(pydantic.BaseModel):
    """Basic MongoDB to SQL conversor for partial push-down of queries."""

    field_pattern: re.Pattern = re.compile(r"[\w.]+")  # Allowed chars in field name in SQL query

    def parse_sorting(self, sorting: Sorting, coll: str) -> Tuple[str, Sorting]:  # noqa
        """Parse MongoDB sorting rule into SQL expression and rule remainder for partial sort push-down."""
        clauses = []
        remainder = []

        for raw_subj, direction in sorting:
            if raw_subj.startswith("$") or not (subj := self._parse_subj(coll, raw_subj)):
                remainder.append((raw_subj, direction))
            else:
                clauses.append(f"{subj} {self._parse_direction(coll, direction)}")

        return ", ".join(clauses), remainder

    @classmethod
    def _parse_direction(cls, coll: str, direction: Direction) -> str:  # noqa
        return {1: "ASC", -1: "DESC"}[direction]

    def parse_query(self, query: Query, coll: str) -> Tuple[str, Query]:
        """Parse MongoDB query into SQL expression and query remainder for partial query push-down."""
        clauses = []
        remainder = {}

        for raw_subj, raw_filter in query.items():
            if raw_subj.startswith("$") or not (subj := self._parse_subj(coll, raw_subj)):
                remainder[raw_subj] = raw_filter
                continue

            if not isinstance(raw_filter, dict):
                raw_filter = {"$eq": raw_filter}

            for raw_pred, raw_obj in raw_filter.items():
                clause_args = (coll, subj, raw_pred, raw_obj)
                if (pred := self._parse_pred(*clause_args)) and (obj := self._parse_obj(*clause_args)):
                    clauses.append(" ".join([subj, pred, obj]))
                else:
                    remainder.setdefault(raw_subj, {})[raw_pred] = raw_obj

        return " AND ".join(clauses), remainder

    def _parse_subj(self, coll: str, subj: str) -> str | None:
        return subj if subj is not None and self.field_pattern.fullmatch(subj) else None

    @classmethod
    def _parse_pred(cls, coll: str, subj: str, raw_pred: Any, raw_obj: Any) -> str | None:  # noqa
        return {
            "$eq": "IS" if raw_obj is None else "=",
            "$gt": ">",
            "$lt": "<",
            "$neq": "!=",
            "$gte": ">=",
            "$lte": "<=",
        }.get(raw_pred)

    @classmethod
    def _parse_obj(cls, coll: str, subj: str, pred: Any, raw_obj: Any) -> str | None:  # noqa
        match raw_obj:
            case None:
                return "NULL"
            case str():
                return "'%s'" % raw_obj.translate(str.maketrans({"'": "''"}))
            case _:
                return str(raw_obj)
