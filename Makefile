install:
	pip install --upgrade pip && pip install -r requirements.txt

test:
	python -m pytest -vv tests/*.py

format:
	black */*.py --line-length=100

lint:
	ruff check ./llamlam

all: install lint test format