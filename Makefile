.PHONY: run-fastapi install run-graph run-graph-dev init

install:
	poetry install

# TODO: 
# run-fastapi: install
# 	poetry run python run.py

run-graph: install
	poetry run streamlit run graph_app.py

run-graph-dev: install
	poetry run streamlit run graph_app.py --server.runOnSave=true

init:
	poetry shell

