from pydantic import BaseModel


class ClassifiedInstrument(BaseModel):
    source_id: str
    instrument: str