import json
import uuid
from kafka import KafkaProducer, KafkaConsumer

BOOTSTRAP = "localhost:9092"
TOPIC_IN = "tmdb_features_in"
TOPIC_OUT = "tmdb_predictions_out"

producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

consumer = KafkaConsumer(
    TOPIC_OUT,
    bootstrap_servers=BOOTSTRAP,
    auto_offset_reset="latest",
    enable_auto_commit=True,
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
)

def send_and_wait():
    request_id = str(uuid.uuid4())

    features = {
        "id": request_id,
        "budget": 12000000,
        "runtime": 110,
        "popularity": 35.2,
        "vote_count": 2400,
        "release_year": 2019,
        "original_language": "en",
    }

    print("➡️ Wysyłam cechy do Kafki:")
    print(features)
    producer.send(TOPIC_IN, features)
    producer.flush()

    print("⏳ Czekam na predykcję...")
    for msg in consumer:
        if msg.value.get("id") == request_id:
            print("✅ Wynik klasyfikacji:")
            print(f"Klasa: {msg.value['prediction']}")
            print(f"Prawdopodobieństwo: {msg.value['probability']:.2f}")
            break

if __name__ == "__main__":
    send_and_wait()
