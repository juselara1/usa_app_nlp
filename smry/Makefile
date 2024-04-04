SHELL := /usr/bin/env bash

install:
	pip install .

install-dev:
	pip install .[dev]

install-playwright:
	playwright install-deps
	playwright install

generate-ssl:
	openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365

update:
	pip install . --no-dependencies

run: update
	streamlit run scripts/app.py

.PHONY: build
build:
	docker compose --env-file env_vars.env build 

deploy:
	docker compose --env-file env_vars.env up
