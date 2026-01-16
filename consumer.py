import json
from kafka import KafkaConsumer

BOOTSTRAP = "localhost:9092"
TOPIC_OUT = "tmdb_predictions_out"

consumer = KafkaConsumer(
    TOPIC_OUT,
    bootstrap_servers=BOOTSTRAP,
    auto_offset_reset="latest",
    enable_auto_commit=True,
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
)

print("Listening for predictions...")
for msg in consumer:
    print("PRED:", msg.value)
