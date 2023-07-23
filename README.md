# track_classifier

web app to help in the manual track classification

## Table of Contents

* [Installation](#installation)
* [Usage](#usage)
* [Testing](#testing)

## Installation

1. After cloning this project, create a `virtualenv` with `python3`

```shell
python3 -m venv .venv
```

2. Activate your `virtualenv`

```shell
source .venv/bin/activate
```

3. Install the dependencies

```shell
pip install -r requirements.txt
```

4. Setup [Google Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials?hl=pt-br)

```shell
gcloud auth application-default login
```

## Testing

1. Run the tests

```shell
pytest
```
