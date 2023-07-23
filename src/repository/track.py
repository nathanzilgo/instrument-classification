from src.model.track import Track


class TrackRepository:
    def find_imported_not_deleted_tracks(self):
        return [
            Track(
                id='98a08171-540d-4943-9fbe-49a6aeaf35a9',
                imported=True,
                video_thumbnail_url='78f419b7-efb1-4ae1-adf8-f8e019bc9b07/98a08171-540d-4943-9fbe-49a6aeaf35a9/video_thumbnail_1e58b88d788dba48bc0a34ac60223a69.jpg',
            ),
            Track(
                id='590f4177-573a-4fe2-850b-fd0345983f49',
                imported=True,
                video_thumbnail_url='78f419b7-efb1-4ae1-adf8-f8e019bc9b07/590f4177-573a-4fe2-850b-fd0345983f49/video_thumbnail_dadd3b6db49d10c5aa1cb448385b6e0e.jpg',
            ),
            Track(
                id='d0c62707-3be2-43cc-88bc-52a2155c5fb8',
                imported=True,
                video_thumbnail_url='78f419b7-efb1-4ae1-adf8-f8e019bc9b07/d0c62707-3be2-43cc-88bc-52a2155c5fb8/video_thumbnail_f5e80fcaaf4380cc587f442b55a3727b.jpg',
            ),
            Track(
                id='da436bd3-c9d4-4c27-bd1f-e978c83e624a',
                imported=True,
                video_thumbnail_url='78f419b7-efb1-4ae1-adf8-f8e019bc9b07/da436bd3-c9d4-4c27-bd1f-e978c83e624a/video_thumbnail_b2ac78cf2441f5919b026b1722000f4b.jpg',
            ),
        ]
