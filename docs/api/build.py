from pathlib import Path

from sphinx.application import Sphinx

source_path = Path(__file__).parent
build_path = source_path / "_build"

Sphinx(
    buildername="html",
    srcdir=str(source_path),
    confdir=str(source_path),
    outdir=str(build_path / "html"),
    doctreedir=str(build_path / ".doctrees"),
).build()
