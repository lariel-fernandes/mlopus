from kedro.framework.session import KedroSession


def test_pipelines(tmp_overrides):
    for pipeline in ["build", "eval"]:
        with KedroSession.create(env="empty", extra_params=tmp_overrides) as session:
            session.run(pipeline)
