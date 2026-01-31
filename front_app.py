import csv
import json
import random
import time
import uuid
from datetime import datetime, timezone
import ast

from kafka import KafkaProducer, KafkaConsumer

BOOTSTRAP = "127.0.0.1:29092"
TOPIC_IN = "tmdb_features_in"
TOPIC_OUT = "tmdb_predictions_out"

CSV_PATH = "data/tmdb_10000_movies.csv"
SEND_EVERY_S = 1.0

producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP,
    acks="all",
    retries=10,
    linger_ms=20,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

consumer = KafkaConsumer(
    TOPIC_OUT,
    bootstrap_servers=BOOTSTRAP,
    auto_offset_reset="latest",
    enable_auto_commit=True,
    group_id="front_app_tmdb",
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    consumer_timeout_ms=300,
)


def now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def to_bool(x):
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in ["true", "1", "t", "yes"]:
        return True
    if s in ["false", "0", "f", "no"]:
        return False
    return None


def parse_release_year(release_date: str):
    if not release_date:
        return None
    try:
        return int(str(release_date).split("-")[0])
    except Exception:
        return None


def parse_genre_count(genre_ids_str: str):
    # genre_ids w CSV wygląda np. "[28, 12, 878]"
    if not genre_ids_str:
        return None
    try:
        arr = ast.literal_eval(genre_ids_str)
        if isinstance(arr, list):
            return int(len(arr))
        return None
    except Exception:
        return None


def load_rows(path: str):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def pick_random_record(rows):
    row = random.choice(rows)
    request_id = str(uuid.uuid4())

    payload = {
        "id": request_id,
        "event_time": now_iso_utc(),

        # meta (dla debug/raportu)
        "tmdb_id": row.get("id"),
        "title": row.get("title"),

        # features zgodne ze streaming schema i batch ML
        "popularity": to_float(row.get("popularity")),
        "vote_count": to_float(row.get("vote_count")),
        "release_year": parse_release_year(row.get("release_date")),
        "original_language": row.get("original_language") or "unknown",
        "adult": to_bool(row.get("adult")),
        "video": to_bool(row.get("video")),
        "genre_count": parse_genre_count(row.get("genre_ids")),
    }
    return payload


def poll_prediction_bundle(expected_id: str, max_wait_s: float = 5.0):
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        for msg in consumer:
            val = msg.value or {}
            if val.get("id") == expected_id:
                return val
        time.sleep(0.05)
    return None


def print_bundle(bundle: dict):
    preds = (bundle.get("predictions") or [])
    preds = sorted(preds, key=lambda x: str(x.get("model", "")))

    print("\n--- PREDICTIONS BUNDLE ---")
    print("id:", bundle.get("id"))
    print("event_time:", bundle.get("event_time"))
    print("processed_time:", bundle.get("processed_time"))
    print("latency_ms:", bundle.get("latency_ms"))

    if not preds:
        print("Brak predictions:", bundle)
        return

    for p in preds:
        print(f"- {p.get('model')}: pred={p.get('prediction')} score={p.get('score')}")


def main():
    rows = load_rows(CSV_PATH)
    if not rows:
        raise RuntimeError(f"CSV pusty albo nie dało się wczytać: {CSV_PATH}")

    print(f"Loaded {len(rows)} rows from {CSV_PATH}")
    print(f"Sending every {SEND_EVERY_S}s to {TOPIC_IN} and waiting on {TOPIC_OUT}...")

    while True:
        payload = pick_random_record(rows)

        print("\nSend:", datetime.now().strftime("%H:%M:%S"), payload)
        producer.send(TOPIC_IN, payload)
        producer.flush()

        bundle = poll_prediction_bundle(payload["id"], max_wait_s=5.0)
        if bundle:
            print_bundle(bundle)
        else:
            print("Brak predykcji w 5s")

        time.sleep(SEND_EVERY_S)


if __name__ == "__main__":
    main()
