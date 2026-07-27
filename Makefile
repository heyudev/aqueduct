SHELL := /bin/bash
.DEFAULT_GOAL := help

# Which model config to launch: inference/vllm/models/<MODEL>.env
MODEL ?= primary
MODEL_ENV := inference/vllm/models/$(MODEL).env

INFRA := docker compose --project-directory inference/vllm -f inference/vllm/docker-compose.yml
GW    := docker compose --project-directory gateway -f gateway/docker-compose.yml --env-file .env

# Model settings are sourced into the shell so docker compose can interpolate
# them; compose only reads the shell environment and the project .env file.
define with_env
set -a; . ./.env; . $(MODEL_ENV); set +a;
endef

.PHONY: help
help: ## Show this help
	@echo "Aqueduct — internal LLM inference service"
	@echo
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Model configs available (pass as MODEL=<name>):"
	@ls inference/vllm/models/*.env 2>/dev/null | xargs -n1 basename 2>/dev/null | sed 's/\.env$$/  /' | sed 's/^/  /' || true

.PHONY: preflight
preflight: ## Check the GPU host is ready (run this first on new hardware)
	@./scripts/preflight.sh

.PHONY: env
env: ## Create .env from the template
	@test ! -f .env || { echo ".env already exists, refusing to overwrite"; exit 1; }
	@cp .env.example .env
	@echo "Created .env — fill in the secrets before starting anything."

# ---- inference layer (GPU machine) ----------------------------------------

.PHONY: up-model
up-model: check-env ## Start vLLM with MODEL=<name> (default: primary)
	@test -f $(MODEL_ENV) || { echo "no such model config: $(MODEL_ENV)"; exit 1; }
	@echo "Starting vLLM with $(MODEL_ENV)"
	@$(with_env) $(INFRA) up -d
	@echo "Loading weights takes several minutes. Follow with: make logs-model"

.PHONY: down-model
down-model: ## Stop vLLM
	@$(with_env) $(INFRA) down

.PHONY: logs-model
logs-model: ## Tail vLLM logs
	@$(with_env) $(INFRA) logs -f --tail=200

.PHONY: wait-model
wait-model: check-env ## Block until vLLM answers /health (up to 30 min)
	@echo -n "Waiting for vLLM"
	@set -a; . ./.env; set +a; \
	for i in $$(seq 1 180); do \
	  if curl -fsS --max-time 5 "http://127.0.0.1:$${VLLM_PORT:-8000}/health" >/dev/null 2>&1; then \
	    echo " — ready"; exit 0; \
	  fi; \
	  echo -n "."; sleep 10; \
	done; \
	echo " — timed out. Check: make logs-model"; exit 1

# ---- gateway layer ---------------------------------------------------------

.PHONY: up-gateway
up-gateway: check-env ## Start LiteLLM proxy and PostgreSQL
	@$(GW) up -d
	@echo "Gateway starting on port 4000."

.PHONY: down-gateway
down-gateway: ## Stop the gateway (data in the pgdata volume is kept)
	@$(GW) down

.PHONY: logs-gateway
logs-gateway: ## Tail gateway logs
	@$(GW) logs -f --tail=200

.PHONY: restart-gateway
restart-gateway: ## Reload gateway config without touching the database
	@$(GW) restart litellm

# ---- operations ------------------------------------------------------------

.PHONY: smoke
smoke: ## End-to-end test through the gateway
	@./scripts/smoke.sh

.PHONY: bench
bench: ## Load test the inference layer directly
	@./scripts/bench.sh

.PHONY: key
key: check-env ## Mint an API key: make key TEAM=search BUDGET=100
	@./scripts/mint-key.sh

.PHONY: usage
usage: check-env ## Show spend per key
	@source ./.env && curl -fsS -H "Authorization: Bearer $$LITELLM_MASTER_KEY" \
	  "$${AQ_URL:-http://127.0.0.1:4000}/spend/keys" | python3 -m json.tool

.PHONY: check-env
check-env:
	@test -f .env || { echo "No .env found. Run: make env"; exit 1; }
