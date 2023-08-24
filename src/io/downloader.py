from urllib.request import urlretrieve

from src.io.local_file import LocalFile


class Downloader:
    def download(self, track) -> LocalFile:
        """
        Downloads a file from the specified URL and returns the local file object.

        Args:
            url (str): The URL of the file to download.

        Returns:
            LocalFile: The local file object representing the downloaded file.
        """
        print(f'>> Downloading "{track.audio_url}"...')
        filename, _ = urlretrieve(track.audio_url, track.id)
        print(f'>> File successfully downloaded: {filename}')
        return LocalFile(path=filename, name=track.id)
