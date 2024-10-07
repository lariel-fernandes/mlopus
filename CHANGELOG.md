## 1.0.1 (2024-10-07)

### Fix

- **typing_utils**: fix typevar and typing alias coercion to type (used in artschema type param inference)
- **kedro**: fix registering extra scopes as config resolvers, fix artschema dataset config parsing in pydantic v2

## 1.0.0 (2024-10-06)

### Refactor

- adopt pydantic v2 (#11)

## 0.3.3 (2024-10-03)

### Fix

- **artschema.framework**: skip warning when there's no dumper conf to be saved
- **kedro.pipeline_factory**: accept mapping instead of config loader
- **mlflow.traits**: allow passing parent run ID to run manager

## 0.3.2 (2024-09-12)

### Fix

- **kedro.hooks.mlflow_tracker**: fix param `prepend_dataset` for mlflow metrics

## 0.3.1 (2024-09-12)

### Fix

- **kedro.hooks.mlflow_tracker**: allow disabling dataset name prefix in metrics
- **kedro.config_loader**: let source default to base, auto-register new scopes as resolvers
- **artschema**: hint to correct format of fq class name
- **artschema**: allow passing str path, auto convert to pathlib

## 0.3.0-2 (2024-09-12)

## 0.3.0 (2024-08-29)

### Feat

- kedro hooks, datasets, config loader and session patch

## 0.2.1 (2024-08-27)

### Fix

- **BaseMlflowApi**: protect cached files, allow duplication by default if using caching from local repo

## 0.2.0 (2024-08-23)

### Feat

- artschema, lineage

## 0.1.1 (2024-08-21)

### Fix

- package metadata

## 0.1.0 (2024-08-21)

### Feat

- base mlflow api and plugins
