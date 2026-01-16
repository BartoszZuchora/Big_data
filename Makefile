SHELL := /bin/bash

# ==========================
# Konfiguracja
# ==========================
VENV_NAME := .venv
PYTHON311 := /opt/homebrew/bin/python3.11
PYTHON := $(VENV_NAME)/bin/python
PIP := $(VENV_NAME)/bin/pip

APP := tmdb_spark_ml.py
SPARK_SUBMIT ?= spark-submit

export PYSPARK_PYTHON := $(abspath $(PYTHON))
export PYSPARK_DRIVER_PYTHON := $(abspath $(PYTHON))

# ==========================
# PHONY
# ==========================
.PHONY: help build python311 venv install run clean destroy check

help:
	@echo ""
	@echo "========================================"
	@echo "  TMDB Big Data – Makefile commands"
	@echo "========================================"
	@echo ""
	@echo "SETUP:"
	@echo "  make build        - install Python 3.11, create venv, install deps"
	@echo "  make check        - show Python and Spark versions"
	@echo ""
	@echo "BATCH (Spark + MLlib):"
	@echo "  make run          - run Spark batch job"
	@echo "  make train        - train ML model and save it"
	@echo ""
	@echo "KAFKA:"
	@echo "  make kafka-up     - start Kafka (Docker)"
	@echo "  make kafka-topics - create Kafka topics"
	@echo "  make kafka-down   - stop Kafka"
	@echo ""
	@echo "STREAMING:"
	@echo "  make stream       - run Spark Structured Streaming"
	@echo ""
	@echo "FRONT:"
	@echo "  make front        - run CLI front app (send + receive prediction)"
	@echo ""
	@echo "CLEANUP:"
	@echo "  make clean        - remove temporary files"
	@echo "  make destroy     - remove venv and all artifacts"
	@echo ""
	@echo "========================================"
	@echo ""

# ==========================
# Python 3.11
# ==========================
python311:
	@if [ ! -x "$(PYTHON311)" ]; then \
		echo ">>> Python 3.11 not found. Installing via Homebrew..."; \
		brew install python@3.11; \
	else \
		echo ">>> Python 3.11 already installed"; \
	fi

# ==========================
# Virtualenv
# ==========================
venv: python311
	@if [ ! -d "$(VENV_NAME)" ]; then \
		echo ">>> Creating venv with Python 3.11"; \
		$(PYTHON311) -m venv $(VENV_NAME); \
	else \
		echo ">>> venv already exists"; \
	fi

# ==========================
# Dependencies
# ==========================
install: venv
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt

build: install

# ==========================
# Run Spark
# ==========================
run:
	@echo ">>> Using PYSPARK_PYTHON=$(PYSPARK_PYTHON)"
	@$(SPARK_SUBMIT) $(APP)

# ==========================
# Checks
# ==========================
check:
	@echo "Python (venv):"
	@$(PYTHON) --version
	@echo ""
	@echo "Spark:"
	@$(SPARK_SUBMIT) --version || true

# ==========================
# Cleanup
# ==========================
clean:
	@rm -rf __pycache__ *.parquet metastore_db spark-warehouse derby.log 2>/dev/null || true

destroy: clean
	@rm -rf $(VENV_NAME)

# ==========================
# Kafka + Streaming + Front
# ==========================

KAFKA_IN=tmdb_features_in
KAFKA_OUT=tmdb_predictions_out

# Dopasowane do Spark 4.1.x
SPARK_KAFKA_PKG ?= org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1

.PHONY: kafka-up kafka-down kafka-topics train stream front

kafka-up:
	docker compose up -d

kafka-down:
	docker compose down

kafka-topics:
	@docker exec -it $$(docker ps -q --filter "name=kafka") \
		kafka-topics --bootstrap-server kafka:29092 \
		--create --if-not-exists \
		--topic $(KAFKA_IN) --partitions 1 --replication-factor 1
	@docker exec -it $$(docker ps -q --filter "name=kafka") \
		kafka-topics --bootstrap-server kafka:29092 \
		--create --if-not-exists \
		--topic $(KAFKA_OUT) --partitions 1 --replication-factor 1

# trening batch + zapis modelu
train:
	$(SPARK_SUBMIT) $(APP)

# Spark Structured Streaming (Kafka -> ML -> Kafka)
stream:
	$(SPARK_SUBMIT) \
	  --driver-memory 10g \
	  --executor-memory 10g \
	  --conf spark.executor.memoryOverhead=2048m \
	  --conf spark.driver.memoryOverhead=2048m \
	  --packages $(SPARK_KAFKA_PKG) \
	  stream_predict.py

# Front aplikacji (wysyła + odbiera)
front:
	$(PYTHON) front_app.py
