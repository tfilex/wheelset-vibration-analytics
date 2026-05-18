PORT ?= 8501
HOST ?= 0.0.0.0
IMAGE ?= bearing-diagnostics-demo

.PHONY: help install test test-verbose test-file smoke check demo mlflow docker-build docker-run

help:
	@printf "Available targets:\n"
	@printf "  make install                    Sync uv environment with dev dependencies\n"
	@printf "  make test                       Run pytest smoke tests\n"
	@printf "  make test-verbose               Run pytest with verbose output\n"
	@printf "  make test-file FILE=<path>      Run one pytest file\n"
	@printf "  make smoke                      Run model self-verification scripts\n"
	@printf "  make check                      Run tests and smoke checks\n"
	@printf "  make demo [PORT=8501]           Run Streamlit demo\n"
	@printf "  make mlflow                     Run local MLflow UI\n"
	@printf "  make docker-build               Build Streamlit demo Docker image\n"
	@printf "  make docker-run [PORT=8501]     Run Streamlit demo Docker container\n"

install:
	uv sync --dev

test:
	uv run pytest

test-verbose:
	uv run pytest -v

test-file:
	@test -n "$(FILE)" || (printf "Usage: make test-file FILE=tests/test_prediction_model.py\n"; exit 1)
	uv run pytest $(FILE)

smoke:
	uv run python src/classification/model.py
	uv run python src/prediction/model.py

check: test smoke

demo:
	uv run --with-requirements requirements.txt streamlit run app.py --server.port=$(PORT) --server.address=$(HOST)

mlflow:
	./run_mlflow.sh

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p $(PORT):8501 $(IMAGE)
