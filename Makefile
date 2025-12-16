install:
	uv pip install -U pip && uv pip install -r requirements.txt

test:
	python -m pytest -vv tests/*.py

smoke-test:
	python -m llamlam.smoke_test

format:
	ruff format .

lint:
	ruff check ./llamlam

all: install lint test format
