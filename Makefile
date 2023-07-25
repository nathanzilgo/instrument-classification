format:
	python -m blue ./src

check.format:
	python -m blue ./src --check

install:
	pip install -r requirements.txt

lint:
	python -m pyflakes ./src

run:
	flask --app src/app --debug run

test:
	pytest --disable-warnings