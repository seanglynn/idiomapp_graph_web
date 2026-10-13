# Local development
install:
	uv sync

test: install
	uv run pytest -q

lint: install
	uv run flake8 idiomagraph tests
	uv run black --check idiomagraph tests

format: install
	uv run black idiomagraph tests

run-graph: install
	uv run python -m idiomagraph.streamlit.app

run-graph-dev: install
	uv run streamlit run idiomagraph/streamlit/app.py --server.runOnSave=true

# Docker commands - interactive by default
docker-start:
	mkdir -p ollama-models logs
	chmod +x docker/docker-entrypoint.sh
	docker compose build
	@echo "Starting IdiomaGraph in interactive mode (http://localhost:8503)..."
	docker compose up

# Run Docker in detached mode for background operation
docker-start-detached:
	mkdir -p ollama-models logs
	chmod +x docker/docker-entrypoint.sh
	docker compose build
	@echo "IdiomaGraph starting in background at http://localhost:8503"
	docker compose up -d

docker-down:
	docker compose down

# Debugging
docker-logs:
	docker compose logs -f

docker-shell:
	docker exec -it idiomagraph-streamlit /bin/bash

ollama-shell:
	docker exec -it idiomagraph-ollama /bin/bash
