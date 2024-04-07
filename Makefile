install:
	pip install --upgrade pip && pip install -r requirements.txt

test:
	python -m pytest -vv *.py

format:
	black */*.py

lint:
	ruff check .

all: install lint test format