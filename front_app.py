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


def poll_predictions_bundle(expected_id: str, max_wait_s: float = 3.0):
    """
    Czeka na 1 wiadomość z wynikami wszystkich modeli:
    {"id": expected_id, "predictions": [...]}.
    """
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        for msg in consumer:
            val = msg.value or {}
            if val.get("id") == expected_id:
                return val
        time.sleep(0.05)
    return None


def print_predictions_bundle(bundle: dict):
    preds = bundle.get("predictions") or []
    if not preds:
        print("Brak pola predictions albo pusta lista. Cały payload:", bundle)
        return

    preds = sorted(preds, key=lambda x: str(x.get("model", "")))

    headers = ["model", "prediction", "score"]
    rows = []
    for p in preds:
        rows.append(
            {
                "model": p.get("model"),
                "prediction": p.get("prediction"),
                "score": p.get("score"),
            }
        )

    col_widths = {
        h: max(len(h), max(len(str(r.get(h))) for r in rows))
        for h in headers
    }

    def fmt_row(r):
        return " | ".join(str(r.get(h)).ljust(col_widths[h]) for h in headers)

    print("\nWyniki wszystkich modeli (porównanie):")
    print(fmt_row({h: h for h in headers}))
    print("-+-".join("-" * col_widths[h] for h in headers))
    for r in rows:
        print(fmt_row(r))


def main():
    rows = load_rows(CSV_PATH)
    if not rows:
        raise RuntimeError(f"CSV pusty albo nie dało się wczytać: {CSV_PATH}")

    print(f"Loaded {len(rows)} rows from {CSV_PATH}")
    print(
        f"Sending one random record every {SEND_EVERY_S}s to {TOPIC_IN} and "
        f"waiting for bundled predictions on {TOPIC_OUT}..."
    )

    while True:
        features, meta = pick_random_features(rows)

        print("\nData wysłania:", datetime.now().strftime("%H:%M:%S"))
        print("meta:", meta)
        print("features:", features)

        producer.send(TOPIC_IN, features)
        producer.flush()

        bundle = poll_predictions_bundle(features["id"], max_wait_s=3.0)
        if bundle:
            print_predictions_bundle(bundle)
        else:
            print("Brak predykcji w 3s")

        time.sleep(SEND_EVERY_S)


if __name__ == "__main__":
    main()