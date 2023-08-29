format:
	python -m blue ./inda_mir ./scripts ./track_classifier_app

check.format:
	python -m blue ./inda_mir ./scripts ./track_classifier_app --check

install:
	pip install -e .

lint:
	python -m pyflakes ./inda_mir ./scripts ./track_classifier_app

run:
	flask --app track_classifier_app/app --debug run

test:
	pytest --disable-warnings

process:
	make delete.output
	python scripts/track_data_cleanse.py

delete.output:
	rm -rf ./output-inda

delete.raw:
	rm -rf ./output-inda/raw
