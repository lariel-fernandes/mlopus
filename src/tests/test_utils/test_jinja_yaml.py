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
            conn_defaults:
              foo: bar
              bar: foo
            with_suffix: {{ "foo" | add_suffix }}
            empty_key: ""
        """)
        )

        (base / "database.yml").write_text(
            dedent("""
            host: {{ common.api_url }}/db
            port: 5432
            timeout: {{ common.timeout }}
            password: {{ env.DB_PASSWORD }}
            api_key: {{ secrets.api_key }}
            missing_gets_fallback: {{ common.missing_key | default("fallback") }}
            empty_gets_fallback: {{ common.empty_key | default("fallback", True) }}
            conn_defaults: {{ common.conn_defaults }}
            conn:
              a: b
              b: a
              {% for key, value in common.conn_defaults.items() %}
              {{ key }}: {{ value }}
              {% endfor %}
        """)
        )

        result = load_jinja_yaml_configs(
            base,
            namespaces=["common", "database"],
            expose_env=True,
            extra_namespaces={"secrets": {"api_key": "abc-xyz-789"}},
            overrides={"common": {"api_url": "https://custom.example.com"}},
            custom_filters={"add_suffix": lambda x: x + "bar"},
        )

        assert result["common"]["api_url"] == "https://custom.example.com"
        assert result["common"]["timeout"] == 30
        assert result["common"]["optional_flag"] is None
        assert result["common"]["optional_env_var"] is None
        assert result["common"]["with_suffix"] == "foobar"
        assert result["database"]["host"] == "https://custom.example.com/db"
        assert result["database"]["port"] == 5432
        assert result["database"]["timeout"] == 30
        assert result["database"]["password"] == "secret123"
        assert result["database"]["api_key"] == "abc-xyz-789"
        assert result["database"]["conn"] == {
            "a": "b",
            "b": "a",
            "foo": "bar",
            "bar": "foo",
        }
        assert result["database"]["missing_gets_fallback"] == "fallback"
        assert result["database"]["empty_gets_fallback"] == "fallback"
        assert result["database"]["conn_defaults"] == result["common"]["conn_defaults"]
