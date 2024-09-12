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
