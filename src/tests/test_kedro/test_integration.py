from kedro.framework.session import KedroSession

import mlopus

pytest_plugins = "tests.test_kedro.fixtures"


class AnyNodeDatasetOrHook(mlopus.mlflow.MlflowRunMixin):
    pass


def test_mlflow_run_mixin_in_session(example_proj, temp_mlflow_overrides):
    overrides = temp_mlflow_overrides

    with KedroSession.create(example_proj, extra_params=overrides) as session:
        config = session.load_context().config_loader

        # An object with `MlflowRunMixin` starts a run with session ID tag
        subj_1 = AnyNodeDatasetOrHook(mlflow=config["globals"]["mlflow"])
        assert subj_1.run_manager.run.status == mlopus.mlflow.RunStatus.RUNNING
        assert subj_1.run_manager.run.tags["kedro"]["active_session"]["uuid"] == session._store["uuid"]
        assert subj_1.run_manager.mlflow_api.cache_dir == overrides["globals"]["mlflow"]["api"]["conf"]["cache_dir"]

        # In the same session, another object finds the same run
        subj_2 = AnyNodeDatasetOrHook(mlflow=config["globals"]["mlflow"])
        assert (run := subj_2.run_manager.run) == subj_1.run_manager.run
        run.end_run()

    overrides["globals"]["mlflow"]["run"] = {"id": run.id}

    with KedroSession.create(example_proj, extra_params=overrides) as session:
        config = session.load_context().config_loader

        # In a new session, an object with `MlflowRunMixin` resumes the run by ID
        subj_3 = AnyNodeDatasetOrHook(mlflow=config["globals"]["mlflow"])
        assert subj_3.run_manager.run.id == run.id
        assert subj_3.run_manager.run.status == mlopus.mlflow.RunStatus.RUNNING
