.PHONY: setup run test

setup:
	@echo "=> Syncing dependencies with uv..."
	uv sync
	@echo "=> Running onboarding wizard..."
	uv run onboard

run:
	uv run python app.py

test:
	uv run pytest
