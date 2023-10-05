import json
import os


def parse_config_file(filepath):
    config = json.load(open(filepath, 'r'))

    if 'ROOT' not in config['dirs']:
        raise Exception

    os.makedirs(config['dirs']['ROOT'], exist_ok=True)

    for dir in config['dirs']:
        if dir == 'ROOT':
            continue
        full_dir = os.path.join(config['dirs']['ROOT'], config['dirs'][dir])
        os.makedirs(full_dir, exist_ok=True)
        config['dirs'][dir] = full_dir

    for metadata in config['metadata']:
        config['metadata'][metadata] = os.path.join(
            config['dirs']['METADATA'], config['metadata'][metadata]
        )

    for output in config['outputs']:
        config['outputs'][output] = os.path.join(
            config['dirs'][output], config['outputs'][output]
        )

    return config


instrument_classification_config = parse_config_file(
    'scripts/config_files/labeling_other_tracks.json'
)
