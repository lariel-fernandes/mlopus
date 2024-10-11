import contextlib
import os
import shutil
from pathlib import Path
from typing import Literal, Iterable, TypeVar

T = TypeVar("T")

PathLike = Path | str

PathOperation = Literal["copy", "move", "link"]


class IllegalPath(Exception):
    """Generic exception for illegal paths."""


class Mode:
    """File permission modes."""

    rwx = 0o770
    r_x = 0o550


def is_sub_dir(x: PathLike, y: PathLike) -> bool:
    """Tell if `x` is subdirectory of `y`."""
    return Path(x).expanduser().resolve().is_relative_to(Path(y).expanduser().resolve())


def is_cwd(path: PathLike) -> bool:
    """Tell whether the path points to the current working directory."""
    return Path(path).expanduser().resolve() == Path.cwd()


def ensure_is_dir(path: PathLike, force: bool = False) -> Path:
    """Ensure that specified path exists and is a directory."""
    if (path := Path(path)).is_file() or is_broken_link(path):
        if force:
            path.unlink()
        raise NotADirectoryError(path)

    path.mkdir(exist_ok=True, parents=True)
    return path


def is_broken_link(path: PathLike):
    """Detect broken symlink."""
    return not (path := Path(path)).exists() and path.is_symlink()


def ensure_non_existing(path: PathLike, force: bool = False) -> Path:
    """Ensure that specified path doesn't exist yet, but its parents do."""
    if (path := Path(path)).is_dir():
        if not force:
            raise IsADirectoryError(path)
        elif path.is_symlink():
            path.unlink()
        else:
            shutil.rmtree(path)
    elif path.is_file() or is_broken_link(path):
        if not force:
            raise FileExistsError(path)
        path.unlink()

    return path


def ensure_empty_dir(path: PathLike, force: bool = False) -> Path:
    """Ensure that specified path is an empty directory."""
    if (path := Path(path)).is_file() or is_broken_link(path) or (path.is_dir() and os.listdir(path)):
        ensure_non_existing(path, force)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_only_parents(path: PathLike, force: bool = False) -> Path:
    """Ensure that specified path doesn't exist yet, but its parents do."""
    ensure_non_existing(path := Path(path), force)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def is_rel_link(path: PathLike) -> bool:
    """Tell if path is relative symbolic link."""
    return (path := Path(path)).is_symlink() and not Path(os.readlink(path)).is_absolute()


def iter_links(path: PathLike) -> Iterable[Path]:
    """Find and iterate symbolic links."""
    if is_rel_link(path := Path(path)):
        yield path
    if path.is_dir():
        for dirpath, dirnames, filenames in os.walk(path):
            for child in dirnames + filenames:
                if (link := Path(dirpath).joinpath(child)).is_symlink():
                    yield link


def place_path(src: PathLike, tgt: PathLike, mode: PathOperation, overwrite: bool = False, move_abs_links: bool = True):
    """Place source file or dir on target using selected operation."""
    src = Path(src)
    ensure_only_parents(tgt := Path(tgt), force=overwrite)

    if mode == "move":
        for link in iter_links(src):
            if (rel := is_rel_link(link)) or not move_abs_links:
                raise RuntimeError(
                    f"Cannot move path '{src}' containing {'relative ' if rel else ''}symbolic link: {link}"
                )
        src.rename(tgt)
    elif mode == "copy":
        shutil.copytree(src, tgt, symlinks=True) if src.is_dir() else shutil.copy(src, tgt, follow_symlinks=True)
    elif mode == "link":
        tgt.symlink_to(src.expanduser().resolve())
    else:
        raise NotImplementedError(f"mode='{mode}'")


def chmod(path: PathLike, mode: int):
    """Apply chmod to file or directory."""
    if (path := Path(path)).is_dir():
        rchmod(path, mode)
    else:
        path.chmod(mode)


def rchmod(path: PathLike, mode: int):
    """Apply recursive chmod to directory."""
    if not (path := Path(path)).is_dir():
        raise NotADirectoryError(path)

    path.chmod(mode)

    for dirpath, dirnames, filenames in os.walk(path):
        for child in dirnames + filenames:
            Path(dirpath).joinpath(child).chmod(mode)


@contextlib.contextmanager
def dir_lock(path: Path):
    """Directory is unlocked inside this context, then protected against modifications on closure."""
    try:
        rchmod(path, Mode.rwx) if path.exists() else None
        yield path
    finally:
        rchmod(path, Mode.r_x) if path.exists() else None


def iter_files(path: PathLike) -> Iterable[Path]:
    """Recursively iterate files in dir yielding their paths relative to that dir."""
    for subdir, _, files in os.walk(path):
        for file in files:
            yield Path(subdir) / file
