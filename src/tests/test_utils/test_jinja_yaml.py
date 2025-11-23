from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent

from mlopus.utils.env_utils import using_env_vars
from mlopus.utils.jinja_yaml import load_jinja_yaml_configs


def test_load_jinja_yaml_configs_basic():
    with TemporaryDirectory() as tmp, using_env_vars({"DB_PASSWORD": "secret123"}):
        base = Path(tmp)

        (base / "common.yml").write_text(
            dedent("""
            api_url: https://api.example.com
            timeout: 30
            optional_flag: null
            optional_env_var: {{ env.OPTIONAL_ENV_VAR }}
        """)
        )

        (base / "database.yml").write_text(
            dedent("""
            host: {{ common.api_url }}/db
            port: 5432
            timeout: {{ common.timeout }}
            password: {{ env.DB_PASSWORD }}
            api_key: {{ secrets.api_key }}
        """)
        )

        result = load_jinja_yaml_configs(
            base,
            namespaces=["common", "database"],
            include_env=True,
            extra_namespaces={"secrets": {"api_key": "abc-xyz-789"}},
            overrides={"common": {"api_url": "https://custom.example.com"}},
        )

        assert result["common"]["api_url"] == "https://custom.example.com"
        assert result["common"]["timeout"] == 30
        assert result["common"]["optional_flag"] is None
        assert result["common"]["optional_env_var"] is None
        assert result["database"]["host"] == "https://custom.example.com/db"
        assert result["database"]["port"] == 5432
        assert result["database"]["timeout"] == 30
        assert result["database"]["password"] == "secret123"
        assert result["database"]["api_key"] == "abc-xyz-789"
