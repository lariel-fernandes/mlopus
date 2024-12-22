import inspect
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List

from mlopus.utils import import_utils, pydantic, urls


class ObjMeta(pydantic.BaseModel):
    Name: str
    Size: int
    IsDir: bool
    MimeType: str
    ModTime: datetime

    @classmethod
    def parse_many(cls, objs: Iterable[pydantic.ModelLike]) -> Iterable["ObjMeta"]:
        for obj in objs:
            yield cls.parse_obj(obj)


LsResult = List[ObjMeta] | ObjMeta


class FileTransfer(pydantic.BaseModel):
    """File transfer wrapper for MLflow API."""

    prog_bar: bool = pydantic.Field(default=True, description="Show progress bar when transfering files.")
    tool: Any = pydantic.Field(
        default="rclone_python.rclone",
        description=(
            "Fully qualified path of module, class or object that exposes the methods/functions "
            "`ls`, `copyto` and `sync`, with signatures compatible with the ones exposed in "
            "`rclone_python.rclone <https://pypi.org/project/rclone-python>`_."
        ),
    )
    extra_args: dict[str, list[str]] = pydantic.Field(
        default={"sync": ["--copy-links"]},
        description="Dict of extra arguments to pass to each of the functions exposed by the :attr:`tool`.",
    )
    use_scheme: str | None = pydantic.Field(
        default=None,
        description="Replace remote URL schemes with this one. Incompatible with :attr:`map_scheme`.",
    )
    map_scheme: dict[str | re.Pattern, str] | None = pydantic.Field(
        default=None,
        description=(
            "Replace remote URL schemes with the first value in this mapping whose key (regexp) matches the URL. "
            "Incompatible with :attr:`use_scheme`."
        ),
    )

    @pydantic.validator("map_scheme")  # noqa
    @classmethod
    def _compile_map_scheme_regex(cls, v: dict | None) -> dict:
        return {re.compile(k) if isinstance(k, str) else k: v for k, v in (v or {}).items()}

    @pydantic.root_validator(mode="after")
    def _scheme_rules_compatibility(self):
        assert False in (bool(self.use_scheme), bool(self.map_scheme)), "`use_scheme` and `map_scheme` are incompatible"
        return self

    @pydantic.root_validator
    def _find_tool(self):
        self.tool = import_utils.find_attr(self.tool) if isinstance(self.tool, str) else self.tool
        return self

    def _translate_scheme(self, url: urls.UrlLike) -> urls.UrlLike:
        if urls.is_local(url):
            return url

        scheme = None
        if self.use_scheme:
            scheme = self.use_scheme
        elif self.map_scheme:
            for pattern, new_scheme in self.map_scheme.items():
                if pattern.match(str(url)):
                    scheme = new_scheme
                    break

        return urls.parse_url(url)._replace(scheme=scheme) if scheme else url  # noqa

    def ls(self, url: urls.UrlLike) -> LsResult:
        """If `url` is a dir, list the objects in it. If it's a file, return the file metadata."""
        objs = list(ObjMeta.parse_many(self._tool("ls", url := str(self._translate_scheme(url)))))

        if len(objs) == 1 and (not (one_obj := objs[0]).IsDir and one_obj.Name == Path(url).name):
            return one_obj

        return objs

    def is_file(self, url: urls.Url) -> bool:
        """Check if URL points to a file. If False, it may be a dir or not exist."""
        return not isinstance(self.ls(url), list)

    def pull_files(self, src: urls.Url, tgt: Path):
        """Pull files from `src` to `tgt`."""
        match self.ls(src):
            case []:
                raise FileNotFoundError(src)
            case list():
                func = "sync"
            case ObjMeta():
                func = "copyto"
            case _:
                raise NotImplementedError("src=%s (%s)", src, type(src))

        src = self._translate_scheme(src)
        self._tool(func, *(str(x).rstrip("/") for x in (src, tgt)))

    def push_files(self, src: Path, tgt: urls.Url):
        """Push files from `src` to `tgt`."""
        tgt = self._translate_scheme(tgt)
        self._tool(
            "copyto" if src.is_file() else "sync",
            *(str(x).rstrip("/") for x in (src.expanduser().resolve(), tgt)),
        )

    def _tool(self, func: str, *args, **kwargs):
        call = getattr(self.tool, func)

        if self.prog_bar and "show_progress" in inspect.signature(call).parameters:
            kwargs["show_progress"] = True

        return call(
            *args,
            **kwargs,
            args=self.extra_args.get(func) or None,
        )
