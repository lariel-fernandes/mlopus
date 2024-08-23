from datetime import datetime
from pathlib import Path
from typing import List, Any, Iterable

from mlopus.utils import urls, pydantic, import_utils


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
    """File transfer tools for MLflow API."""

    prog_bar: bool = True
    tool: Any = "rclone_python.rclone"

    @pydantic.root_validator  # noqa
    @classmethod
    def _find_tool(cls, values: dict) -> dict:
        if isinstance(tool := values.get("tool"), str):
            values["tool"] = import_utils.find_attr(tool)
        return values

    def ls(self, url: urls.UrlLike) -> LsResult:
        """If `url` is a dir, list the objects in it. If it's a file, return the file metadata."""
        objs = list(ObjMeta.parse_many(self._tool("ls", url := str(url))))

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

        self._tool(func, *(str(x).rstrip("/") for x in (src, tgt)))

    def push_files(self, src: Path, tgt: urls.Url):
        """Push files from `src` to `tgt`."""
        self._tool(
            "copyto" if src.is_file() else "sync",
            *(str(x).rstrip("/") for x in (src.expanduser().resolve(), tgt)),
        )

    def _tool(self, func: str, *args, **kwargs):
        return getattr(self.tool, func)(*args, **kwargs)