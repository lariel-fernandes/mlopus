#!make

# === MLflow Sandbox ==============================================================================

$(addprefix mlflow-sandbox-,%): ../../docker/mlflow-sandbox/
	@cd $< && $(MAKE) -s $(@:mlflow-sandbox-%=%) REL_PATH=$<
