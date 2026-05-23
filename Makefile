PORT ?= 8501
HOST ?= 0.0.0.0
IMAGE ?= bearing-diagnostics-demo
MODEL_MODE ?= demo
MODEL_MODE_LOCKED ?= 1

.PHONY: help install test test-verbose test-file smoke check demo mlflow vkr-materials docker-build docker-run docker-run-demo docker-run-experimental

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
	@printf "  make vkr-materials              Generate VKR tables and figures from saved CSV metrics\n"
	@printf "  make docker-build               Build Streamlit demo Docker image\n"
	@printf "  make docker-run [MODEL_MODE=demo|experimental] [PORT=8501]\n"
	@printf "  make docker-run-demo [PORT=8501]           Run locked semi-production demo\n"
	@printf "  make docker-run-experimental [PORT=8501]   Run locked experimental demo\n"

install:
	@printf "[install] Syncing uv environment with dev dependencies\n"
	@uv sync --dev

test:
	@printf "[test] Running pytest\n"
	@uv run pytest

test-verbose:
	@printf "[test] Running pytest with verbose output\n"
	@uv run pytest -v

test-file:
	@test -n "$(FILE)" || (printf "Usage: make test-file FILE=tests/test_prediction_model.py\n"; exit 1)
	@printf "[test] Running pytest file: $(FILE)\n"
	@uv run pytest $(FILE)

smoke:
	@printf "[smoke] Running classification model self-check\n"
	@uv run python src/classification/model.py
	@printf "\n[smoke] Running RUL model self-check\n"
	@uv run python src/prediction/model.py

check: test smoke

demo:
	@printf "[demo] Starting Streamlit on $(HOST):$(PORT)\n"
	@uv run --with-requirements requirements.txt streamlit run app.py --server.port=$(PORT) --server.address=$(HOST)

mlflow:
	@printf "[mlflow] Starting local MLflow UI\n"
	@./run_mlflow.sh

vkr-materials:
	@printf "[vkr] Generating VKR tables and figures from saved CSV metrics\n"
	@uv run python scratch_scripts/make_vkr_materials.py

docker-build:
	@printf "[docker] Building image $(IMAGE)\n"
	@docker build -t $(IMAGE) .

docker-run:
	@printf "[docker] Running image $(IMAGE) on port $(PORT) with MODEL_CATALOG_MODE=$(MODEL_MODE)\n"
	@docker run --rm -p $(PORT):8501 \
		-e MODEL_CATALOG_MODE=$(MODEL_MODE) \
		-e MODEL_CATALOG_LOCKED=$(MODEL_MODE_LOCKED) \
		$(IMAGE)

docker-run-demo:
	@$(MAKE) docker-run MODEL_MODE=demo MODEL_MODE_LOCKED=1 PORT=$(PORT)

docker-run-experimental:
	@$(MAKE) docker-run MODEL_MODE=experimental MODEL_MODE_LOCKED=1 PORT=$(PORT)
