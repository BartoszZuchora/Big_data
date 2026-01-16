import json
import uuid
import time
from kafka import KafkaProducer, KafkaConsumer

BOOTSTRAP = "127.0.0.1:29092"  # bezpieczniej niż localhost (IPv6 potrafi mieszać)
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
    consumer_timeout_ms=15000,  # żeby nie wisiał wiecznie
)

def send_and_wait(timeout_s: int = 15):
    request_id = str(uuid.uuid4())

    features = {
        "id": request_id,
        "budget": 12000000.0,
        "runtime": 110.0,
        "popularity": 4900.2,
        "vote_count": 2400.0,
        "release_year": 2019,
        "original_language": "en",
    }

    print("➡️ Wysyłam cechy do Kafki:")
    print(features)
    producer.send(TOPIC_IN, features)
    producer.flush()

    print("⏳ Czekam na predykcję...")
    t0 = time.time()

    for msg in consumer:
        val = msg.value or {}
        if val.get("id") == request_id:
            print("✅ Wynik klasyfikacji:")
            print(f"Klasa: {val.get('prediction')}")
            prob = val.get("probability")
            if isinstance(prob, (int, float)):
                print(f"Prawdopodobieństwo (label=1): {prob:.2f}")
            else:
                print(f"Prawdopodobieństwo (label=1): {prob}")
            return

        if time.time() - t0 > timeout_s:
            break

    raise TimeoutError("Nie dostałem predykcji w czasie (sprawdź czy Spark streaming działa).")

if __name__ == "__main__":
    send_and_wait()
