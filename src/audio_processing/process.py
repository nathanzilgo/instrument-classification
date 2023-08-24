import logging

from sample_operation import SampleOperation
from silence_filter import SilenceFilter
from src.io.downloader import Downloader
from src.io.local_file import LocalFile
from src.app.repository.track_repository import TrackRepository


def process():
    """
    Process tracks by applying a SilenceFilter and a SampleOperation.

    This function is responsible for processing tracks by applying a SilenceFilter
    and a SampleOperation. It takes no parameters.

    Returns:
        dict: A dictionary with a single key 'message' and a value of 'OK' if the
        operation is successful.

    Raises:
        Exception: If any exception occurs during the operation, an exception is raised
        with a message indicating the cause of the failure.
    """
    tracks = TrackRepository().find_labelled_tracks()
    downloader = Downloader()
    logging.basicConfig(level=logging.INFO)

    try:
        for track in tracks:
            logging.info(f'Processing track {track.id}')

            downloaded: LocalFile = downloader.download(track)

            output_path = f'./output/{track.id}'

            silenced = SilenceFilter.apply(
                downloaded, None, 'ogg', output_path, 100, -45, 30
            )
            SampleOperation.apply(
                silenced, None, 'ogg', 10000, output_path, False
            )

            logging.info(
                f"""Processed track {track.id},
                        no errors occurred \n
                        Samples saved at ./output/""",
            )
        return {'message': 'OK'}

    except Exception:
        logging.error(logging.ERROR, f'Error on track {track.id}')
        raise Exception('Operation failed')


if __name__ == '__main__':
    process()
