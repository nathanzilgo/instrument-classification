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

extract:
	python scripts/feature_extraction.py

upload:
	python scripts/bucket_upload.py

process:
	make clean
	python scripts/track_data_cleanse.py

clean:
	rm -rf ./output-inda

clean.raw:
	rm -rf ./output-inda/raw

lightgbm.mac:
	brew install lightgbm
	pip install lightgbm

lightgbm.linux:
	git clone --recursive https://github.com/microsoft/LightGBM
	cd LightGBM
	mkdir build
	cd build
	cmake ..
	make -j4
