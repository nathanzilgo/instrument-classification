FROM debian11
FROM python:3.11.5

ADD requirements.txt /

ADD Makefile / scripts/ setup.py / inda_mir/ / output-inda/ /

RUN apt-get install build-essential libeigen3-dev libyaml-dev libfftw3-dev libavcodec-dev libavformat-dev libavutil-dev libswresample-dev libsamplerate0-dev libtag1-dev libchromaprint-dev

RUN apt-get install python3-dev python3-numpy-dev python3-numpy python3-yaml python3-six

RUN make install.linux

RUN make extract

