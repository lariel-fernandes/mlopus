
MlflowTracker
=============

.. automodule:: mlopus.kedro.hooks.mlflow_tracker

.. autopydantic_model:: mlopus.kedro.MlflowTracker
   :exclude-members: state, factory, after_catalog_created, after_context_created, after_pipeline_run, before_dataset_loaded, before_dataset_saved, before_pipeline_run, on_pipeline_error
   :member-order: bysource
   :inherited-members:

.. autopydantic_model:: mlopus.kedro.hooks.mlflow_tracker.Report
   :member-order: bysource

.. autopydantic_model:: mlopus.kedro.hooks.mlflow_tracker.Logs
   :member-order: bysource

.. autopydantic_model:: mlopus.kedro.hooks.mlflow_tracker.LogFile
   :member-order: bysource
   :exclude-members: parse_obj

.. autopydantic_model:: mlopus.kedro.hooks.mlflow_tracker.Tags
   :member-order: bysource

.. autopydantic_model:: mlopus.kedro.hooks.mlflow_tracker.TagsMlflow
   :exclude-members: apply, apply_suffix, apply_prefix
   :member-order: bysource
   :inherited-members:

.. autopydantic_model:: mlopus.kedro.hooks.mlflow_tracker.Metrics
   :exclude-members: apply,
   :member-order: bysource

.. autopydantic_model:: mlopus.kedro.hooks.mlflow_tracker.MetricsMlflow
   :exclude-members: apply, apply_suffix, apply_prefix
   :member-order: bysource
   :inherited-members:

.. autopydantic_model:: mlopus.kedro.hooks.mlflow_tracker.Params
   :member-order: bysource

.. autopydantic_model:: mlopus.kedro.hooks.mlflow_tracker.ParamsMlflow
   :exclude-members: apply, apply_suffix, apply_prefix
   :member-order: bysource
   :inherited-members:

.. autopydantic_model:: mlopus.kedro.hooks.mlflow_tracker.Overrides
   :inherited-members:
   :exclude-members: apply, apply_suffix, apply_prefix, allow, process_key, process_val
   :member-order: bysource

.. autopydantic_model:: mlopus.kedro.hooks.mlflow_tracker.Config
   :inherited-members:
   :exclude-members: apply, apply_suffix, apply_prefix, allow, process_key, process_val
   :member-order: bysource

.. autopydantic_model:: mlopus.kedro.hooks.mlflow_tracker.ConfigMlflow
   :inherited-members:
   :exclude-members: apply, apply_suffix, apply_prefix, allow, process_key, process_val
   :member-order: bysource

.. autopydantic_model:: mlopus.kedro.hooks.mlflow_tracker.Datasets
   :inherited-members:
   :exclude-members: apply, allow, process_key, process_val
   :member-order: bysource

.. autopydantic_model:: mlopus.kedro.hooks.mlflow_tracker.DatasetsMlflow
   :inherited-members:
   :exclude-members: apply, apply_suffix, apply_prefix, allow, process_key, process_val
   :member-order: bysource

.. autopydantic_model:: mlopus.kedro.hooks.mlflow_tracker.Nodes
   :inherited-members:
   :exclude-members: apply, allow, process_key, process_val
   :member-order: bysource

.. autopydantic_model:: mlopus.kedro.hooks.mlflow_tracker.NodesMlflow
   :inherited-members:
   :exclude-members: apply, apply_suffix, apply_prefix, allow, process_key, process_val
   :member-order: bysource

.. autopydantic_model:: mlopus.kedro.hooks.mlflow_tracker.Rule
   :inherited-members:
   :exclude-members: allow, match, process_subject
   :member-order: bysource

.. autopydantic_model:: mlopus.kedro.hooks.mlflow_tracker.NodeRule
   :inherited-members:
   :exclude-members: allow, match, process_subject
   :member-order: bysource
