from unittest import mock

from track_classifier_app.model.track import Track


def test_home(client):
    with mock.patch(
        'track_classifier_app.app.TrackRepository'
    ) as track_repository_mock:
        track_repository_instance_mock = mock.Mock()
        track_repository_instance_mock.find_imported_not_deleted_tracks.return_value = [
            Track(
                id='id1', video_url='video1', audio_url='audio1', imported=True
            ),
            Track(
                id='id2', video_url='video2', audio_url='audio2', imported=True
            ),
            Track(
                id='id3', video_url='video3', audio_url='audio3', imported=True
            ),
            Track(
                id='id4', video_url='video4', audio_url='audio4', imported=True
            ),
        ]

        track_repository_mock.return_value = track_repository_instance_mock

        response = client.get('/')
        assert response.status_code == 200
