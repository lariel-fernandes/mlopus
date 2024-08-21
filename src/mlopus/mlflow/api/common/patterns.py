import re
import urllib.parse

EXP_ID = re.compile(r"[\w\-]+")
RUN_ID = re.compile(r"[\w\-]+")
MODEL_NAME = re.compile(r"[\w\-./]+")
MODEL_VERSION = re.compile(r"[\w\-.]+")
TAG_PARAM_OR_METRIC_KEY = re.compile(r"[\w\-/]+")


def encode_model_name(name: str) -> str:
    """Encode model name to be used in URL or Path.

    From the special chars allowed in the `MODEL_NAME` pattern,
    only the slash (/) is excluded from safe chars.
    """
    return urllib.parse.quote(name, safe="_-.")
