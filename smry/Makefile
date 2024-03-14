SHELL := /usr/bin/env bash

install:
	pip install .

install-dev:
	pip install .[dev]

install-playwright:
	playwright install-deps
	playwright install

update:
	pip install . --no-dependencies

run: update
	streamlit run scripts/app.py

.PHONY: build
build:
	docker compose --env-file env_vars.env build 

deploy:
	docker compose --env-file env_vars.env up
