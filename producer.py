import json
import time
import uuid
from kafka import KafkaProducer

BOOTSTRAP = "localhost:9092"
TOPIC_IN = "tmdb_features_in"

producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

def send_one():
    msg = {
        "id": str(uuid.uuid4()),
        "budget": 12000000,
        "runtime": 110,
        "popularity": 35.2,
        "vote_count": 2400,
        "release_year": 2019,
        "original_language": "en",
    }
    producer.send(TOPIC_IN, msg)
    producer.flush()
    print("Sent:", msg)

if __name__ == "__main__":
    while True:
        send_one()
        time.sleep(2)
