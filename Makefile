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
	python scripts/remove_track_silence_and_sample.py
