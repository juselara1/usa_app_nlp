SHELL := /usr/bin/env bash

install:
	pip install .

install-dev:
	pip install .[dev]

update:
	pip install . --no-dependencies

run: update
	streamlit run scripts/app.py --server.port=8000 --server.address=0.0.0.0

.PHONY: build
build:
	docker compose --env-file env_vars.env build 

deploy:
	docker compose --env-file env_vars.env up
