from unittest import mock


def test_home(client):
    with mock.patch('src.app.TrackRepository') as track_repository_mock:
        track_repository_instance_mock = mock.Mock()
        track_repository_instance_mock.find_imported_not_deleted_tracks.return_value = [
            ['id1', 'path1'],
            ['id2', 'path2'],
            ['id3', 'path3'],
            ['id4', 'path4'],
        ]

        track_repository_mock.return_value = track_repository_instance_mock

        response = client.get('/')
        assert response.status_code == 200
