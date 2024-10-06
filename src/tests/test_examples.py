import re
from pathlib import Path
from typing import Callable

import nbformat
from kedro.framework.session import KedroSession
from nbconvert.preprocessors import ExecutePreprocessor

pytest_plugins = "tests.test_kedro.fixtures"


def _edit_cell(nb: nbformat.NotebookNode, pos: int, func: Callable[[str], str]) -> None:
    cell = nb["cells"][pos]
    cell["source"] = func(cell["source"])


def test_example_1(temp_mlflow):
    root = Path("examples/1_introduction")
    variables = {"tracking_uri": temp_mlflow.tracking_uri, "cache_dir": temp_mlflow.cache_dir}

    for name in ("Part-1.ipynb", "Part-2.ipynb"):
        print("Testing notebook:", name)
        notebook = nbformat.read(root / name, as_version=4)

        for key, val in variables.items():
            # In cell 2 (API setup) replace variables
            _edit_cell(notebook, 2, lambda x: re.sub(rf'"{key}": .*', f'"{key}": "{val}",', x))

        # In cell 5 (Env setup in Part-2) replace `pip install` with `sys append`, so we don't need a notebook restart
        _edit_cell(notebook, 4, lambda x: re.sub(r"!.*pip.* +install.* +(.*)", "import sys; sys.path.append('\\1')", x))

        ExecutePreprocessor(timeout=60).preprocess(notebook, resources={"metadata": {"path": root}})


def test_example_2(temp_dir, example_proj, temp_mlflow_overrides):
    overrides = temp_mlflow_overrides
    overrides["globals"]["model"] = {"version": "1"}

    with temp_dir() as tmp:
        overrides["globals"]["data"] = tmp / "data"  # Use a temp data dir

        for pipeline in ("build", "eval"):
            print("Testing pipeline:", pipeline)

            with KedroSession.create(example_proj, extra_params=temp_mlflow_overrides) as session:
                session.run(pipeline)
