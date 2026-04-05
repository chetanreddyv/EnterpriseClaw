.PHONY: setup run test

setup:
	@echo "=> Syncing dependencies with uv..."
	uv sync --extra setup
	@echo "=> Running onboarding wizard..."
	uv run --extra setup onboard

run:
	uv run python app.py

test:
	uv run pytest
