from src.app.model.track import Track
from google.cloud import bigquery


class TrackRepository:

    VIDEO_BASE_PATH = 'https://uploads.storage.inda.band/'
    AUDIO_BASE_PATH = 'https://uploads.storage.inda.band/'

    def label_summary(self):
        query = """
        SELECT
            lb.label AS instrument,
            COUNT(lb.track_id) AS count,
            SUM(t.duration_in_milliseconds)/1000 AS duration_in_sec
        FROM
            track_classification.track_labels lb,
            inda_api.tracks t
        WHERE
            lb.track_id = t.id
        GROUP BY
            label;
        """
        client = bigquery.Client()
        query_job = client.query(query)
        return [dict(row) for row in query_job]

    def insert_labels(self, tracks):
        client = bigquery.Client()
        query = """INSERT INTO track_classification.track_labels VALUES\n"""

        for track in tracks['labels']:
            for lb in track['label']:
                query += f"('{track['track_id']}', '{lb}'),\n"

        query = query[:-2] + ';'
        query_job = client.query(query)
        query_job.result()

        return

    def get_tracks_by_query(self, query):
        client = bigquery.Client()

        query_job = client.query(query)  # Make an API request.

        tracks = []
        for row in query_job:
            tracks.append(
                Track(
                    id=row[0],
                    video_url=self.VIDEO_BASE_PATH + row[1]
                    if 'https' not in row[1]
                    else row[1],
                    audio_url=self.AUDIO_BASE_PATH + row[2]
                    if 'https' not in row[2]
                    else row[2],
                    imported=True,
                )
            )

        return tracks

    def find_imported_not_deleted_tracks(self):

        query = """
            SELECT id, video_url, audio_url
            FROM inda_api.tracks
            WHERE deleted_at is null and created_at > '2023-04-20' and instrument='' and audio_url!='' and id not in (
                SELECT track_id FROM track_classification.track_labels
            )
            ORDER BY created_at desc
            LIMIT 10
        """

        return self.get_tracks_by_query(query)

    def find_imported_not_deleted_tracks_by_user(self, uname):

        query = f"""
            SELECT t.id, t.video_url, t.audio_url
            FROM inda_api.tracks t, inda_api.users u
            WHERE t.deleted_at is null and instrument='' and t.user_id=u.id and u.username = '{uname}' and audio_url!='' and id not in (
                SELECT track_id FROM track_classification.track_labels
            )
            ORDER BY t.created_at desc
            LIMIT 10
        """

        return self.get_tracks_by_query(query)

    def find_imported_not_deleted_tracks_by_instrument(self, instrument):

        query = f"""
            SELECT id, video_url, audio_url
            FROM inda_api.tracks
            WHERE deleted_at is null and created_at > '2023-04-20' and instrument='{instrument}' and audio_url!='' and id not in (
                SELECT track_id FROM track_classification.track_labels
            )
            ORDER BY created_at desc
            LIMIT 10
        """

        return self.get_tracks_by_query(query)
