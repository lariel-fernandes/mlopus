def _resolve_pipeline_name(run_params: dict) -> str:
    if isinstance(pipeline_name := run_params.get("pipeline_name"), str):
        return pipeline_name

    if isinstance(pipeline_names := run_params.get("pipeline_names"), list):
        assert len(pipeline_names) == 1, f"Expected exactly one pipeline name, got {pipeline_names}"
        return pipeline_names[0]

    raise ValueError(f"Could not resolve pipeline name from run parameters: {run_params}")
