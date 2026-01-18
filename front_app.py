import csv
import json
import random
import time
import uuid
from datetime import datetime

from kafka import KafkaProducer, KafkaConsumer

BOOTSTRAP = "127.0.0.1:29092"
TOPIC_IN = "tmdb_features_in"
TOPIC_OUT = "tmdb_predictions_out"

CSV_PATH = "data/tmdb_10000_movies.csv"
SEND_EVERY_S = 1.0

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
    consumer_timeout_ms=200,
)

def parse_release_year(release_date: str):
    if not release_date:
        return None
    try:
        return int(release_date.split("-")[0])
    except Exception:
        return None

def to_float(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "null":
        return None
    try:
        return float(s)
    except Exception:
        return None

def load_rows(path: str):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]

def pick_random_features(rows):
    row = random.choice(rows)

    request_id = str(uuid.uuid4())

    features = {
        "id": request_id,

        "popularity": to_float(row.get("popularity")),
        "vote_count": to_float(row.get("vote_count")),

        "release_year": parse_release_year(row.get("release_date")),
        "original_language": (row.get("original_language") or "unknown"),
    }

    meta = {
        "tmdb_id": row.get("id"),
        "title": row.get("title"),
        "vote_average": row.get("vote_average"),
    }
    return features, meta

def poll_predictions(expected_id: str, max_wait_s: float = 2.0):
    """Czyta predykcje przez max_wait_s i zwraca wynik dla expected_id, jeśli się pojawi."""
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        for msg in consumer:
            val = msg.value or {}
            if val.get("id") == expected_id:
                return val
        time.sleep(0.05)
    return None

def main():
    rows = load_rows(CSV_PATH)
    if not rows:
        raise RuntimeError(f"CSV pusty albo nie dało się wczytać: {CSV_PATH}")

    print(f"Loaded {len(rows)} rows from {CSV_PATH}")
    print(f"Sending one random record every {SEND_EVERY_S}s to {TOPIC_IN} and waiting for {TOPIC_OUT}...")

    while True:
        features, meta = pick_random_features(rows)

        print("\nData wysłania: ", datetime.now().strftime("%H:%M:%S"))
        print("meta:", meta)
        print("features:", features)

        producer.send(TOPIC_IN, features)
        producer.flush()

        pred = poll_predictions(features["id"], max_wait_s=2.0)
        if pred:
            print("Wynik predykcji: ", pred)
        else:
            print("Brak predykcji w 2s")

        time.sleep(SEND_EVERY_S)

if __name__ == "__main__":
    main()