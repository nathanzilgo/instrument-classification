# TCC: Musical Instrument Classification using labeled isolated tracks: an analysis involving public datasets and the influence of this pipeline techniques.

## IRMAS Dataset Experiments
IRMAS is intended to be used for training and testing methods for the automatic recognition of predominant instruments in musical audio. The instruments considered are: cello, clarinet, flute, acoustic guitar, electric guitar, organ, piano, saxophone, trumpet, violin, and human singing voice. This dataset is derived from the one compiled by Ferdinand Fuhrmann in his PhD thesis (http://www.dtic.upf.edu/~ffuhrmann/PhD/), with the difference that we provide audio data in stereo format, the annotations in the testing dataset are limited to specific pitched instruments, and there is a different amount and lenght of excerpts.

1. For IRMAS training dataset, we will be focusing on evaluating our model with brass for the data in these classes:
   * clarinet (brass)
   * flute (brass)
   * electric guitar (guitar)
   * organ (keyboards)
   * piano (keyboards)
   * saxophone (brass)
   * trumpet (brass)
   * human singing voice (vocals)

2. Then, we can test our model with IRMAS testing datasets 

3. Then, we can test training the model with IRMAS train dataset using our pipeline directives and testing with IRMAS testing datasets, then comparing the performance using or not our pre processing

## Musical Instrument's Sound Dataset Experiments
This Dataset has 4 types of classes .

- Guitar_Sound - 700 Sounds in Train Set
- Drum_Souund - 700 Sounds in Train Set
- Violin_Sound - 700 Sound in Train Set
- Piano_Sound - 528 Sound in Train Set

In Test Set Total 80 audio files are present , 20 From Each Class.

1. For this dataset, we can test our inda mir model removing bass and brass classes and evaluate both on their train data and test data.
2. Also testing our model with their data and compare the results, testing both on their test data and our test data

## Freesound Loop Dataset Experiments (discarded for now)
This dataset contains 9,455 loops from [Freesound.org](http://freesound.org/) and the corresponding annotations. These loops have tempo, key, genre and instrumentation annotation.

**Dataset Construction**

To collect this dataset, the following steps were performed:

*   Freesound was queried with "loop" and "bpm", so as to collect loops which have a beats-per-minute(BPM) annotations.

*   The sounds were analysed with AudioCommons extractor, so as to obtain key information.

*   The textual metadata of each sound was analysed, to obtain the BPM proposed by the user, and to obtain genre information.

*   Annotators used a [web interface](http://mtg.upf.edu/fslannotator/) to annotate around 3,000 loops.

**Dataset Organisation**

The dataset contains two folders and two files in the root directory:

*   'FSL10K' encloses the audio files and their metadata and analysis. The audios are in the 'audio' folder and are named '<freesound_sound_id>.wav'. The AudioCommons analysis of the loops is present in the 'ac_analysis' directory, while the Essentia analysis of the loops obtained through the Freesound API is on the 'fs_analysis' directory. The textual metadata for each audio can be found in the 'metadata.json'. Finally, the audio analysis provided by the algorithms which were benchmarked in the paper is on the 'benchmark' directory.

*   'annotations' holds the expert provided annotation for the sounds in the dataset. The annotations are separated in a folder for each annotator and each annotation is stored as a .json file, named 'sound-<freesound_sound_id>.json', with a key for each of the features extracted.


### Other Datasets to Analyse:
* [Minst](https://github.com/ejhumphrey/minst-dataset)
* [GoodSounds](https://paperswithcode.com/dataset/goodsounds)
* [Slakh](http://www.slakh.com/)

### Articles and reference:
* [Solos: A Dataset for Audio-Visual Music Analysis](https://arxiv.org/pdf/2006.07931.pdf)