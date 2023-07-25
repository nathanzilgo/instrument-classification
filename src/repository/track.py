from src.model.track import Track
from google.cloud import bigquery


class TrackRepository:
    def find_imported_not_deleted_tracks(self):
        client = bigquery.Client()

        query = """
            SELECT id, video_thumbnail_url
            FROM inda_api.tracks
            WHERE deleted_at is null and imported=true
            and created_at > '2023-04-20'
            order by created_at desc
            LIMIT 20
        """
        query_job = client.query(query)  # Make an API request.

        tracks = []
        for row in query_job:
            tracks.append(
                Track(id=row[0], imported=True, video_thumbnail_url=row[1])
            )

        return tracks
