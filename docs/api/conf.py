# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from typing import Mapping

from pydantic.v1 import BaseModel

sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "MLOpus"
author = "Lariel Fernandes"
release = "0.2.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "enum_tools.autoenum",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.autodoc_pydantic",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

language = "en"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Options for autodoc --------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration

autodoc_member_order = "groupwise"

# -- Options for pydantic_autodoc --------------------------------------------
# https://autodoc-pydantic.readthedocs.io/en/stable/users/configuration.html

autodoc_pydantic_model_show_json = False
autodoc_pydantic_model_member_order = "groupwise"
autodoc_pydantic_model_show_field_summary = False
autodoc_pydantic_model_show_config_summary = False
autodoc_pydantic_model_show_validator_summary = False

# -- Event handlers ----------------------------------------------------------


def skip_member_handler(app, objtype, membername, member, skip, options):  # noqa
    if not skip and options.get("inherited-members"):
        for base in [BaseModel, Mapping]:
            if membername not in dir(base):
                continue

            frame = sys._getframe()  # noqa
            while frame.f_code.co_name != "filter_members":
                frame = frame.f_back
            suspect = frame.f_locals["self"].object

            if issubclass(suspect, base):
                return True

    return skip


def setup(app):
    app.connect("autodoc-skip-member", skip_member_handler)
