from pydantic import BaseModel


class Track(BaseModel):
    id: str
    video_url: str
    audio_url: str
    imported: bool
    audio_imported: bool = True
