
from .track import TrackRepository


def test_find_imported_not_deleted_tracks():
    repository = TrackRepository()
    
    result = repository.find_imported_not_deleted_tracks()

    assert len(result) == 4