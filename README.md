# track_classifier

## Table of Contents

- [track\_classifier](#track_classifier)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Configuration file](#configuration-file)
    - [Download, Preprocess, Feature-extraction and Data Partition](#download-preprocess-feature-extraction-and-data-partition)
    - [Sharing your data](#sharing-your-data)
  - [Running the applications](#running-the-applications)
    - [Manual track classifier](#manual-track-classifier)
  - [Testing](#testing)

## Installation

1. After cloning this project, create a `virtualenv` with `python3`

```shell
make venv
```

2. Install the dependencies

```shell
make install.linux
```

or

```shell
make install.mac
```

3. Setup [Google Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials?hl=pt-br)

```shell
gcloud auth application-default login
```

## Usage

### Configuration file

Our scripts use a [configuration file](scripts/config_files/instrument_classification.json) that holds all the configuration needed to run them.

### Download, preprocess, feature extraction and data partition

1. You can download tracks by running:

```shell
make download_tracks
```

This will perform the query specified in the configuration file (`params > download_tracks > QUERY`) and download the tracks returned. This query must return a table with the `audio_url` column. This url must be the path (in the GCS) from where the query will be download. You can configure the bucket from where the tracks will be download in the configuration file (`params > download_tracks > BUCKET_NAME`). The downloaded tracks will be saved at the path specified in `dirs > RAW_TRACKS` and the metadata will be saved at the path specified in `metadata > RAW_TRACKS`.

2. You can process your tracks by running:

```shell
make process_tracks
```

This will run the processing script that breaks the tracks in samples and filter tracks which the silence time is superior to a threshold. The params can be configured at `params > process_tracks`. It will also generate a metadata file listing the samples generated (w/o the silent ones) at `metadata > PROCESSED_SAMPLES`.

3. To perform the feature extraction you must run:

```shell
make feature_extraction
```

The outputs will be saved at the path specified at `outputs > FEATURES`.

4. To split your data in train and test you must run:

```shell
make split
```

The result of the split will be saved at the path specified at `outputs > TRAIN_TEST_SPLITS`.

### Sharing your data

You can download and upload data to our bucket in the GCS, using:

```shell
python scripts/gcs_interface.py [-h] [-o {upload,download}] -t {raw,samples,features,tts,model} [-f FILENAME]
```

The use cases are detailed in the following table:

|               Task              |    -o (operation)   |    -t  (type)  |                                                -f                 (filename)                               |
|-------------------------------|:--------:|:--------:|------------------------------------------------------------------------------------------------|
| Upload a set of tracks from ROOT:RAW_TRACKS         |  upload  |    raw   | Name of the zip that will be generated and uploaded to the GCS.                                  |
| Upload a set of samples from ROOT:PROCESSED_SAMPLES       |  upload  |  samples | Name of the zip that will be generated and uploaded to the GCS.                                  |
| Upload a dataset of features    |  upload  | features | Name of the dataset of features located at ROOT:FEATURES                                         |
| Upload a train/test partition   |  upload  |    tts   | Name of the train/test partition located at ROOT:TRAIN_TEST_SPLITS                               |
| Upload a trained model          |  upload  |   model  | Name of the trained model located at './models'                                                  |
| Download a set of tracks        | download |    raw   | Name of the remote file to be downloaded from BUCKET:raw_tracks to ROOT:RAW_TRACKS               |
| Download a set of samples       | download | samples  | Name of the remote file to be downloaded from BUCKET:samples to ROOT:PROCESSED_SAMPLES           |
| Download a dataset of features  | download | features | Name of the remote file to be downloaded from BUCKET:features to ROOT:FEATURES                   |
| Download a train/test partition | download |    tts   | Name of the remote file to be downloaded from BUCKET:train_test_splits to ROOT:TRAIN_TEST_SPLITS |
| Download a trained model        | download |   model  | Name of the remote file to be downloaded from BUCKET:models to './models'                        |

The uppercased names refer to directories named in the configurations file. The `-o upload` can be ommited since it is the default operation. After downloading a model or a train/test partition to load and use it in your code, use the corresponding functions exported by `inda_mir.loaders`.

## Running the applications

### Manual track classifier

1. Run the application

```shell
make run
```

2. Go to 'http://127.0.0.1:5000' in your browser.

3. There you should see the tracks and buttons with the name of the instruments (see image below).

![Alt text](docs/images/interface.png)

4. Select the instruments that are present in the track (you can select multiple instruments).

5. Scroll down the page and click the 'Submit!' button. 

6. Wait for the alert indicanting the submission is done.

7. Alternatively you can go to:
 - 'http://127.0.0.1:5000/user/[username]' to see tracks of a single user.
 - 'http://127.0.0.1:5000/instrument/[instrument]' to see tracks originated by the track separation feature for a single instrument.

## Testing

1. Run the tests

```shell
pytest
```