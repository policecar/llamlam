install:
	uv pip install -U pip && uv pip install -r requirements.txt

test:
	python -m pytest -vv -m "not slow" tests/

format:
	ruff format .

lint:
	ruff check .
	ruff format --check .

all: install lint test format
