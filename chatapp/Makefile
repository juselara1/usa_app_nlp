SHELL := /usr/bin/env bash

printsmt:
	echo "Starting"

install: printsmt
	pip install .

install-dev:
	pip install .[dev]

run:
	source env_vars.env && streamlit run scripts/chat.py --server.port=8000 --server.address=0.0.0.0
