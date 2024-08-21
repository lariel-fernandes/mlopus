#!make
SHELL := /bin/bash

install: install-project install-tools install-git-hooks

# === UV ==========================================================================================

install-project:
	@uv sync --frozen --all-extras

install-tools:
	@uv tool install -q ruff
	@uv tool install -q build
	@uv tool install -q twine
	@uv tool install -q pre-commit
	@uv tool install -q commitizen
	@uv tool install -q coverage-badge

# === Git hooks ===================================================================================

install-git-hooks:
	@uvx pre-commit install -t pre-commit -t commit-msg -t post-checkout -t post-merge 1>/dev/null

pre-commit: install-git-hooks
	.git/hooks/pre-commit

_hook-install:
	@if [[ "$(PRE_COMMIT_CHECKOUT_TYPE)" != "0" ]] ; then $(MAKE) -s install; fi

# === Commitizen ==================================================================================

commit: pre-commit
	uvx --from commitizen cz commit

# === Pytest ======================================================================================

test:
	uv run pytest -v $(PYTEST_OPTS) ./src/tests  # some pytest settings are defined in pyproject.toml
	@uvx coverage-badge -q -f -o ./htmlcov/coverage.svg

# === Ruff ========================================================================================

ruff-check:
	uvx ruff format --check  # ruff settings are defined in pyproject.toml
	uvx ruff check

ruff-apply:
	uvx ruff format  # ruff settings are defined in pyproject.toml
	uvx ruff check --fix

# === Sphinx ======================================================================================

api-docs:
	cd docs/api && uv run $(MAKE) -s html

# === MLflow Sandbox ==============================================================================

$(addprefix mlflow-sandbox-,%): docker/mlflow-sandbox
	@cd $< && $(MAKE) -s $(@:mlflow-sandbox-%=%) REL_PATH=$<
