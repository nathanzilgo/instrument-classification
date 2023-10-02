format:
	python -m blue ./inda_mir ./scripts ./track_classifier_app ./models_showcase

check.format:
	python -m blue ./inda_mir ./scripts ./track_classifier_app --check

install:
	pip install -e .

install.mac:
	make lightgbm.mac
	make essentia.mac
	pip install -e .
	
install.linux:
	make lightgbm.linux
	pip install -e .
	
lint:
	python -m pyflakes ./inda_mir ./scripts ./track_classifier_app

run:
	flask --app track_classifier_app/app --debug run

run.models_showcase:
	flask --app models_showcase/app --debug run

test:
	pytest --disable-warnings

extract:
	python scripts/feature_extraction.py ${args}

split:
	python scripts/train_test_split.py

upload:
	python scripts/bucket_upload.py

process:
	make clean
	python scripts/track_data_cleanse.py ${args}

query:
	python scripts/util/get_query_metadata.py

clean:
	rm -rf ./output-inda/sampled ./output-inda/silenced 

clean.raw:
	rm -rf ./output-inda/raw

clean.parameterized:
	rm -rf ./output-inda/parameterized

venv:
	python -m venv ./venv && source ./venv/bin/activate

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
