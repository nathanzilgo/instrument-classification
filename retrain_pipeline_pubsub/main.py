import logging
from datetime import datetime, timezone
from typing import Any, Callable
from pydantic import BaseModel
from google.cloud import pubsub
from google.cloud.pubsub_v1 import SubscriberClient
from google.cloud.pubsub_v1.types import FlowControl

from retrain_pipeline_pubsub.src.retrain_model import retrain_model
from retrain_pipeline_pubsub.src.settings import settings


class Response(BaseModel):
    current_date: str = (
        datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    )


def ack_up_message(message: Any) -> None:
    message.ack()
    logging.info('Acking up message. VM will scale down.')


def process_retraining(message: Any) -> None:
    message.ack()
    try:
        logging.info('Starting model retrain')
        retrain_model()
        logging.info('Model retrained successfuly!')

        publisher = pubsub.PublisherClient()
        topic_path = publisher.topic_path(
            settings.PROJECT_ID, settings.FINISH_TOPIC_ID
        )

        future = publisher.publish(
            topic_path, Response().json().encode('utf-8')
        )

        if future.result():
            logging.info(
                f'Published the results to "{topic_path}". Now finishing the pipeline'
            )
            subscribe(
                callback=ack_up_message,
                subscription_id=settings.UP_SUBSCRIPTION_ID,
            )

    except Exception as e:
        logging.info(f'>>>> EXCEPTION, {e}')
        logging.exception(e)


def subscribe(callback: Callable[[Any], Any], subscription_id: str) -> None:
    subscriber = SubscriberClient()
    subscription_path = subscriber.subscription_path(
        settings.PROJECT_ID, subscription_id
    )

    flow_control = FlowControl(max_messages=settings.MAX_MESSAGES)
    streaming_pull_future = subscriber.subscribe(
        subscription_path,
        callback=callback,
        flow_control=flow_control,
    )

    logging.info(f'Listening for messages on {subscription_path}...\n')

    with subscriber:
        try:
            streaming_pull_future.result()
        except TimeoutError:
            streaming_pull_future.cancel()
