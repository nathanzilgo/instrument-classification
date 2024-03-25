from datetime import datetime, timezone
from google.cloud import pubsub
from pydantic import BaseModel

from src.settings import settings

publisher = pubsub.PublisherClient()
topic_path = publisher.topic_path(
    settings.PUBSUB_PROJECT_ID, settings.PUBSUB_FINISH_TOPIC_ID
)


class Response(BaseModel):
    current_date: str = (
        datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    )


response = publisher.publish(
    topic_path,
    Response().json().encode('utf-8'),
)

print(f'Published "{response.result()}" to {topic_path}.')
