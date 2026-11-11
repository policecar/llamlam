.PHONY: install test format lint all

install:
	uv pip install -U pip && uv pip install -r requirements.txt

test:
	python -m pytest -vv tests/

format:
	ruff format llamlam tests

lint:
	ruff check llamlam tests
	ruff format --check llamlam tests

all: install lint test format
