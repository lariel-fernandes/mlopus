#!make

install: install-project install-tools

# === UV ==========================================================================================

install-project:
	@uv sync --frozen --all-extras

install-tools:
	@uv tool install -q coverage-badge

# === Pytest ======================================================================================

test:
	uv run pytest -v $(PYTEST_OPTS) ./src/tests  # some pytest settings are defined in pyproject.toml
	@uvx coverage-badge -q -f -o ./htmlcov/coverage.svg

# === MLflow Sandbox ==============================================================================

$(addprefix mlflow-sandbox-,%): ../../docker/mlflow-sandbox/
	@cd $< && $(MAKE) -s $(@:mlflow-sandbox-%=%) REL_PATH=$<
