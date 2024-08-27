#!make
include .env
export $(shell sed 's/=.*//' .env)

restart:
	docker compose --env-file .env up --wait --quiet-pull --force-recreate --always-recreate-deps --renew-anon-volumes
	@echo "\n###### MLflow Sandbox is running ######"
	@echo "\nAccess the WebUI at:\n\t http://localhost:$(MLFLOW_HOST_PORT)"
	@echo "\nConnect to environment with:\n\t source $(REL_PATH)env.sh"
	@echo "\nRestart server with:\n\t make mlflow-sandbox-restart"
	@echo "\nStop server with:\n\t make mlflow-sandbox-stop"

stop:
	docker compose --env-file .env down
# check that unrelated changes do not trigger tests
