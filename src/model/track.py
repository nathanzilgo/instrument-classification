from pydantic import BaseModel


class Track(BaseModel):
    id: str
    imported: bool
    audio_imported: bool = True
    video_thumbnail_url: str