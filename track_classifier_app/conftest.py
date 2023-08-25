import pytest


@pytest.fixture
def app():
    from track_classifier_app.app import app

    return app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def runner(app):
    return app.test_cli_runner()
