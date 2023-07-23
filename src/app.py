from flask import Flask, render_template

from src.repository.track import TrackRepository

app = Flask(__name__)


@app.route('/')
def main():
    tracks = TrackRepository().find_imported_not_deleted_tracks()
    return render_template('main.html', tracks=tracks)
