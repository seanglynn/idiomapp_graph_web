# Local development
install:
	uv sync

test: install
	uv run pytest -q

lint: install
	uv run flake8 idiomapp tests
	uv run black --check idiomapp tests

format: install
	uv run black idiomapp tests

run-graph: install
	uv run python -m idiomapp.streamlit.app

run-graph-dev: install
	uv run streamlit run idiomapp/streamlit/app.py --server.runOnSave=true

# Docker commands - interactive by default
docker-start:
	mkdir -p ollama-models logs
	chmod +x docker/docker-entrypoint.sh
	docker compose build
	@echo "Starting IdiomApp in interactive mode (http://localhost:8503)..."
	docker compose up

# Run Docker in detached mode for background operation
docker-start-detached:
	mkdir -p ollama-models logs
	chmod +x docker/docker-entrypoint.sh
	docker compose build
	@echo "IdiomApp starting in background at http://localhost:8503"
	docker compose up -d

docker-down:
	docker compose down

# Debugging
docker-logs:
	docker compose logs -f

docker-shell:
	docker exec -it idiomapp-streamlit /bin/bash

ollama-shell:
	docker exec -it idiomapp-ollama /bin/bash
