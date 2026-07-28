.PHONY: help dev lint test clean

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-9s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

dev: ## Install development dependencies and prek
	@uv sync --all-groups
	@uv tool install prek --quiet
	@uvx prek install

lint: ## Run linting, formatting, and type checks
	@uv run --frozen ruff check src/multisusie_cli tests/test_application_models.py tests/test_preparation.py
	@uv run --frozen ruff format --check src/multisusie_cli tests/test_application_models.py tests/test_preparation.py
	@uv run --frozen ty check src/multisusie_cli

test: ## Run the test suite
	@uv run --frozen pytest -rxs

clean: ## Remove local build and test artifacts
	@rm -rf .venv .pytest_cache .ruff_cache *.egg-info
