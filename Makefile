# corpus-rag command surface. Recipes use real tabs.
.DEFAULT_GOAL := help
.PHONY: help setup ingest index serve eval test lint

help: ## Show this help.
	@echo "Targets:"
	@echo "  setup    uv sync (install deps)"
	@echo "  ingest   PDF=<path>  ingest one PDF into outputs/<slug>/ (OUTPUT_LAYOUT)"
	@echo "  index    build the hybrid index from outputs/ (the recipe)"
	@echo "  serve    run the MCP server (stdio)"
	@echo "  eval     run the local-vs-global eval to decide if a graph is needed"
	@echo "  test     pytest"
	@echo "  lint     ruff check ."

setup: ## Install dependencies with uv.
	uv sync

ingest: ## Ingest one PDF: make ingest PDF=path/to/file.pdf
	uv run python -m corpus_rag.ingest.pipeline $(PDF)

index: ## Build the hybrid index from outputs/ per rag.config.yaml.
	uv run python -m corpus_rag.index.build

serve: ## Run the MCP server.
	uv run python -m corpus_rag.server.mcp_server

eval: ## Decide whether a concept graph is warranted (BenchmarkQED-lite).
	uv run python -m corpus_rag.eval.benchmark

test: ## Run tests.
	uv run pytest -q

lint: ## Lint.
	uv run ruff check .
