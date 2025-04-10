.PHONY: run-fastapi install run-graph run-graph-dev init

install:
	poetry install

run-fastapi: install
	poetry run python run.py

run-graph: install
	poetry run streamlit run streamlit_app.py

run-graph-dev: install
	poetry run streamlit run streamlit_app.py --server.runOnSave=true

init:
	poetry shell

