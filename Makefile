.PHONY: install run-fastapi run-graph run-graph-dev init install-dev

install:
	poetry install

run-fastapi: install
	poetry run python -m idiomapp.api.app

run-graph: install
	poetry run python -m idiomapp.streamlit.app

run-graph-dev: install
	poetry run streamlit run idiomapp/streamlit/app.py --server.runOnSave=true

install-dev: install
	poetry install --with dev

init:
	poetry shell

