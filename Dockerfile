# Use the official Python image.
# https://hub.docker.com/_/python
FROM python:3.11-buster

# Copy local code to the container image.
ENV APP_HOME /app
ENV PYTHONUNBUFFERED TRUE

WORKDIR $APP_HOME

# Install ffmpeg
RUN apt-get -y update && apt-get install -y ffmpeg


COPY requirements.txt setup.py ./

COPY inda_mir ./inda_mir

COPY retrain_pipeline_pubsub ./retrain_pipeline_pubsub

COPY scripts ./scripts

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install -e .

CMD ["python3", "retrain_pipeline_pubsub/main.py"]