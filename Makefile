# ==========================
# Konfiguracja
# ==========================
VENV_NAME=.venv
PYTHON=$(VENV_NAME)/bin/python
PIP=$(VENV_NAME)/bin/pip
SPARK_SUBMIT=spark-submit

APP=tmdb_spark_ml.py
DATA=data/tmdb_10000_movies.csv

# ==========================
# Domyślna komenda
# ==========================
.PHONY: help
help:
	@echo "Dostępne komendy:"
	@echo "  make venv        - utworzenie virtualenv"
	@echo "  make install     - instalacja zależności"
	@echo "  make build       - venv + zależności"
	@echo "  make run         - uruchomienie Spark job"
	@echo "  make clean       - usunięcie plików tymczasowych"
	@echo "  make destroy     - USUNIĘCIE venv i buildów"

# ==========================
# Virtualenv
# ==========================
.PHONY: venv
venv:
	python3 -m venv $(VENV_NAME)

.PHONY: install
install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

.PHONY: build
build: venv install

# ==========================
# Uruchomienie Sparka
# ==========================
.PHONY: run
run:
	$(SPARK_SUBMIT) $(APP)

# ==========================
# Sprzątanie
# ==========================
.PHONY: clean
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf *.parquet

.PHONY: destroy
destroy: clean
	rm -rf $(VENV_NAME)
