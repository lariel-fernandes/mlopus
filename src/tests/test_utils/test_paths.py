import os
from pathlib import Path
from typing import Callable, Dict

import pytest

from mlopus.utils import paths
from ..fixtures.utils import random_file, md5_file, md5_dir, raises_if


class TestPlacePath:
    @pytest.fixture
    def src_file(self, temp_dir) -> Path:
        with temp_dir() as tmp:
            random_file(file := tmp.joinpath("file"))
            yield file

    @pytest.fixture
    def src_dir(self, temp_dir) -> Path:
        with temp_dir() as tmp:
            random_file((dir_ := tmp.joinpath("dir")).joinpath("file"))
            yield dir_

    @pytest.fixture(params=["file", "dir"])
    def src(self, request):
        return request.getfixturevalue("src_%s" % request.param)

    @pytest.fixture
    def md5(self, src: Path) -> Callable[[Path], str | Dict[str, str]]:
        return md5_file if src.is_file() else md5_dir

    @pytest.fixture(params=["existing", "new"])
    def existing(self, request) -> bool:
        return request.param == "existing"

    @pytest.fixture(params=["overwrite", "no-overwrite"])
    def overwrite(self, request) -> bool:
        return request.param == "overwrite"

    @pytest.fixture
    def tgt(self, temp_dir, existing: bool) -> Path:
        with temp_dir() as tmp:
            tgt_ = tmp.joinpath("tgt")
            if existing:
                tgt_.touch()
            yield tgt_

    def test_copy(self, md5, src: Path, tgt: Path, existing: bool, overwrite: bool):
        with raises_if(existing and not overwrite, FileExistsError):
            paths.place_path(src, tgt, mode="copy", overwrite=overwrite)
            assert md5(src) == md5(tgt)

    def test_link(self, md5, src: Path, tgt: Path, existing: bool, overwrite: bool):
        with raises_if(existing and not overwrite, FileExistsError):
            paths.place_path(src, tgt, mode="link", overwrite=overwrite)
            assert md5(src) == md5(tgt)
            assert os.readlink(tgt) == str(src)

    def test_move(self, md5, src: Path, tgt: Path, existing: bool, overwrite: bool):
        expected = md5(src)
        with raises_if(existing and not overwrite, FileExistsError):
            paths.place_path(src, tgt, mode="move", overwrite=overwrite)
            assert expected == md5(tgt)
            assert not src.exists()

    def test_move_error_on_rel_links(self, src_dir):
        src_dir.joinpath("rel_link").symlink_to("./file")
        with pytest.raises(RuntimeError):
            paths.place_path(src_dir, Path("./test"), mode="move")

    def test_move_error_on_abs_links(self, src_dir):
        src_dir.joinpath("rel_link").symlink_to(src_dir.joinpath("file").absolute())
        with pytest.raises(RuntimeError):
            paths.place_path(src_dir, Path("./test"), mode="move", move_abs_links=False)
