from src.app.repository.track_repository import TrackRepository
from unittest import mock


@mock.patch('src.app.repository.track_repository.bigquery')
def test_find_imported_not_deleted_tracks(bigquery_mock):
    client_mock = mock.Mock()
    client_mock.query.return_value = [
        ['id1', 'path1', 'path1'],
        ['id2', 'path2', 'path1'],
        ['id3', 'path3', 'path1'],
        ['id4', 'path4', 'path1'],
    ]

    bigquery_mock.Client.return_value = client_mock
    repository = TrackRepository()

    result = repository.find_imported_not_deleted_tracks()

    assert len(result) == 4
