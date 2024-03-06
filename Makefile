# Formatting commands

format:
	python -m blue ./inda_mir ./scripts ./track_classifier_app ./models_showcase ./retrain_pipeline_pubsub

check.format:
	python -m blue ./inda_mir ./scripts ./track_classifier_app --check

lint:
	python -m pyflakes ./inda_mir ./scripts ./track_classifier_app

# Testing commands

test:
	pytest --disable-warnings

# Installation

venv:
	python -m venv ./venv && source ./venv/bin/activate

install:
	pip install -e .

install.mac:
	make lightgbm.mac
	make essentia.mac
	pip install -e .
	
install.linux:
	make lightgbm.linux
	pip install -e .

lightgbm.mac:
	brew install lightgbm

lightgbm.linux:
	sudo apt update && sudo apt upgrade
	sudo apt install cmake
	git clone --recursive https://github.com/microsoft/LightGBM
	cd LightGBM
	mkdir build
	cd build
	cmake ..
	make -j4

essentia.mac:
	git clone --recursive https://github.com/MTG/essentia
	cd essentia
	python waf configure --build-static --with-python --with-cpptests --with-examples 
	python waf install
	python waf
	pip install essentia

# Processing

download_tracks:
	python scripts/download_tracks.py

process_tracks:
	python scripts/process_tracks.py

feature_extraction:
	python scripts/feature_extraction.py

fine_tune:
	python scripts/fine_tuning.py

flaml_fine_tune:
	python scripts/flaml_fine_tuning.py

split:
	python scripts/train_test_split.py

gcs:
	python scripts/gcs_interface.py

retrain:
	python scripts/retrain_model.py

## Applications

run_manual_classifier:
	flask --app track_classifier_app/app --debug run

run_models_showcase:
	flask --app models_showcase/app --debug run