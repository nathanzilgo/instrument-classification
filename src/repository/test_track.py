from .track import TrackRepository
from unittest import mock


@mock.patch('src.repository.track.bigquery')
def test_find_imported_not_deleted_tracks(bigquery_mock):
    client_mock = mock.Mock()
    client_mock.query.return_value = [
        ['id1', 'path1'],
        ['id2', 'path2'],
        ['id3', 'path3'],
        ['id4', 'path4'],
    ]

    bigquery_mock.Client.return_value = client_mock
    repository = TrackRepository()

    result = repository.find_imported_not_deleted_tracks()

    assert len(result) == 4
