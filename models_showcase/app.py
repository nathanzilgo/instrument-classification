from collections import Counter
import os
import shutil
import pandas as pd
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from inda_mir.audio_processing.sample_operation import SampleOperation
from inda_mir.audio_processing.silence_filter import SilenceFilter

from inda_mir.modeling.models import load_model
from inda_mir.modeling.feature_extractor import FreesoundExtractor

app = Flask(__name__)

# Directory containing model files and JSON parameter files
model_directory = 'models/'


def get_model_params(model_name):
    model_file = os.path.join(model_directory, model_name + '.pkl')
    model = load_model(model_file)
    return model, model.get_params()


@app.route('/spec', methods=['GET', 'POST'])
def index():
    model_files = [
        f.split('.')[0]
        for f in os.listdir(model_directory)
        if f.endswith('.pkl')
    ]
    selected_model = None
    model_params = {}

    if request.method == 'POST':
        selected_model = request.form['model_name']
        _, model_params = get_model_params(selected_model)

    return render_template(
        'spec.html',
        model_files=model_files,
        selected_model=selected_model,
        params=model_params,
    )


@app.route('/', methods=['GET', 'POST'])
def test_models():
    OUTPUT_DIR_SILENCE = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'uploads', 'silenced'
    )
    OUTPUT_DIR_SAMPLE = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'uploads', 'sampled'
    )
    OUTPUT_FORMAT = 'ogg'
    SAMPLE_DURATION = 10000

    if request.method == 'POST':
        uploaded_file = request.files['audio_file']
        if uploaded_file.filename != '':
            audio_filename = os.path.join(
                'uploads', secure_filename(uploaded_file.filename)
            )
            audio_filename = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), audio_filename
            )
            uploaded_file.save(audio_filename)

            # Process the audio (add your preprocessing code here)
            audio_url_format = audio_filename.split('.')[-1]
            audio_output_filename = audio_filename.split('/')[-1]

            output_filename = os.path.join(
                OUTPUT_DIR_SILENCE, audio_output_filename
            )
            os.makedirs(OUTPUT_DIR_SILENCE, exist_ok=True)
            output_silence_path = SilenceFilter.apply(
                audio_filename,
                audio_url_format,
                OUTPUT_FORMAT,
                output_filename,
            )

            output_filename = os.path.join(
                OUTPUT_DIR_SAMPLE, audio_output_filename
            )
            os.makedirs(OUTPUT_DIR_SAMPLE, exist_ok=True)
            SampleOperation.apply(
                output_silence_path,
                OUTPUT_FORMAT,
                OUTPUT_FORMAT,
                SAMPLE_DURATION,
                output_filename,
            )

            # Initialize a dictionary to store model predictions
            model_predictions = {}
            fe = FreesoundExtractor()

            # Iterate through each model
            model_files = [
                f.split('.')[0]
                for f in os.listdir(model_directory)
                if f.endswith('.pkl')
            ]

            sample_features = []
            for sample_path in os.scandir(OUTPUT_DIR_SAMPLE):
                sample_path = os.path.abspath(sample_path)

                features = fe.features_to_df(
                    fe._extract(sample_path), file_path=''
                )
                features = features.drop(
                    ['filename', 'frame'], axis=1
                ).to_numpy()
                sample_features.append(features)

            for model_name in model_files:
                model, _ = get_model_params(model_name)
                model_label_predictions = []

                for features in sample_features:

                    # Apply the model to the audio (add your preprocessing code here)
                    predicted_label = model.predict(features)[0]

                    model_label_predictions.append(predicted_label)

                counter = Counter(model_label_predictions)
                predicted_label_voting = counter.most_common(1)[0][0]
                model_predictions[model_name] = predicted_label_voting

            # Create a DataFrame to display model predictions
            df = pd.DataFrame(
                list(model_predictions.items()),
                columns=['Model', 'Predicted Label'],
            )

            os.remove(audio_filename)
            shutil.rmtree(OUTPUT_DIR_SILENCE)
            shutil.rmtree(OUTPUT_DIR_SAMPLE)

            return render_template('index.html', df=df)

    return render_template('index.html', df=None)


if __name__ == '__main__':
    app.run(debug=True, port=8000)
