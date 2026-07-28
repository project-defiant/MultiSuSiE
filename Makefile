.PHONY: help dev lint test smoke docker-smoke clean

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-9s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

dev: ## Install development dependencies and prek
	@uv sync --all-groups
	@uv tool install prek --quiet
	@uvx prek install

lint: ## Run linting, formatting, and type checks
	@uv run --frozen ruff check src/multisusie_cli tests/test_application_models.py tests/test_preparation.py tests/test_runner.py tests/test_anndata_output.py tests/test_study_locus_output.py
	@uv run --frozen ruff format --check src/multisusie_cli tests/test_application_models.py tests/test_preparation.py tests/test_runner.py tests/test_anndata_output.py tests/test_study_locus_output.py
	@uv run --frozen ty check src/multisusie_cli

test: ## Run the test suite
	@uv run --frozen pytest -rxs

smoke: ## Run the synthetic CLI smoke test
	@uv run --frozen python scripts/smoke_test.py

docker-smoke: ## Build the image and run the synthetic smoke test inside it
	@docker build --tag multisusie:smoke .
	@docker run --rm multisusie:smoke --help
	@docker run --rm --entrypoint "" multisusie:smoke multisusie --help
	@docker run --rm --entrypoint uv multisusie:smoke run --no-dev python scripts/smoke_test.py

clean: ## Remove local build and test artifacts
	@rm -rf .venv .pytest_cache .ruff_cache *.egg-info
