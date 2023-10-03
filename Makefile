install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black --verbose *.py utils/*.py

lint:
	pylint --disable=R,C,E1120 *.py utils/*.py
	
# test: python -m pytest -vv --cov=main --cov=nlplogic test_*.py

all: install format lint