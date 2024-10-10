install:
	pip install --upgrade pip && pip install -r requirements.txt

test:
	python -m pytest -vv *.py

format:
	black llamlam/*.py

lint:
	ruff check ./llamlam

all: install lint test format