from flask import Flask, render_template, request
import logging

from track_classifier_app.repository.track_repository import TrackRepository


app = Flask(__name__)

FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)


@app.route('/')
def main():
    tracks = TrackRepository().find_imported_not_deleted_tracks()
    return render_template('main.html', tracks=tracks)


@app.route('/label/', methods=['POST'])
def label():
    tracks = request.get_json()
    TrackRepository().insert_labels(tracks)
    return {'message': 'OK'}


@app.route('/summary', methods=['GET'])
def summary():
    lbsummary = TrackRepository().label_summary()
    return render_template('summary.html', summary=lbsummary)


@app.route('/user/<uname>', methods=['GET'])
def filter_user(uname):
    tracks = TrackRepository().find_imported_not_deleted_tracks_by_user(uname)
    return render_template('main.html', tracks=tracks)


@app.route('/instrument/<instrument>', methods=['GET'])
def filter_instrument(instrument):
    tracks = TrackRepository().find_imported_not_deleted_tracks_by_instrument(
        instrument
    )
    return render_template('main.html', tracks=tracks)
