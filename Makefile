# Local development
install:
	poetry install

run-graph: install
	poetry run python -m idiomapp.streamlit.app

run-graph-dev: install
	poetry run streamlit run idiomapp/streamlit/app.py --server.runOnSave=true

# Docker commands
docker-start:
	mkdir -p ollama-models logs
	docker-compose build
	docker-compose up

docker-down:
	docker-compose down

# Debugging
docker-shell:
	docker exec -it idiomapp-streamlit /bin/bash

ollama-shell:
	docker exec -it idiomapp-ollama /bin/bash
