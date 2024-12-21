## 1.2.0 (2024-12-21)

### Feat

- scheme translation for artifact urls

## 1.1.0 (2024-12-21)

### Feat

- pull artifacts in offline mode (if at least metadata is cached)

### Fix

- enum json encoding

## 1.0.8 (2024-12-02)

### Fix

- **kedro.NodeFunc**: let users implement either __call__ or async __acall__
- **providers/mlflow**: use http client settings in healthcheck, stringify tags/params before max length check, fix parsing local urls

## 1.0.7 (2024-10-25)

### Fix

- **kedro.hooks.mlflow-artifacts**: allow pipeline inptus/outputs to be a mapping
- **artschema.tags**: forbid range constraint on required pkg version if there's build metadata (e.g.: 1.2.3+hash)

## 1.0.6 (2024-10-24)

### Fix

- **artschema.tags**: account for recursive extras when checking requirements of artschema

## 1.0.5 (2024-10-23)

### Fix

- **artschema.spec**: fix entity api property in LoadArtifactSpec

## 1.0.4 (2024-10-23)

### Fix

- **artschema.tags**: fix detection of installed extras and pkg spec verification
- **artschema.spec**: expose entity api in LoadArtifactSpec

## 1.0.3 (2024-10-21)

### Fix

- **utils.dicts**: preserve empty dict leaf vals in deep merge (used in kedro conf merge)
- **kedro.session**: use only root package name for determining version

## 1.0.2 (2024-10-11)

### Fix

- **mlflow.api.transfer**: apply prog bar flag, allow extra args, copy links by default
- **kedro.hooks.artifacts**: fix file permissions after input setup
- **artschema**: skip saving dumper conf if empty, adjust spec parser for pydantic v2
- **utils.pydantic**: coerce int to str

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
