# Overview of MLOpus Architecture

## MLflow APIs via MLOpus
[<img src="/docs/images/mlflow_api.jpg" height="350">]()

Users can interact with the MLOpus MLflow API either directly or through the module `mlopus.artschema`,
which is a higher level abstraction for exchanging run and model artifacts under a pre-defined IO logic.

The module `mlopus.mlflow.api` knows how to perform the basic tasks of an experiment tracking and model registry
system, like managing metadata and artifacts inside the local cache and exchanging artifacts with the cloud storage.

When it comes to communicating with the MLflow server, it relies on plugins. Plugins must implement the interface
`BaseMlflowApi` by handling the exchange of metadata with the MLflow backend that they're built for.
The default plugin is `mlopus.mlflow.providers.MlflowApi`, which works with open source MLflow.

The dotted lines in the diagram above represent operations that are unavailable when the API is
configured in offline mode (i.e.: cache-only mode). This is especially convenient for testing and building
applications that need to run isolated, like model serving apps in which the artifacts cache is populated at build time
or mounted as a Docker volume.

### Constraints
The diagram above implies the following constraints. These constraints are by-design in MLOpus
and should be appropriate for a production MLflow setup.

1. CRUD operations on metadata are only performed via HTTP calls to the MLflow API server.
   This means that the MLflow server database may remain hidden behind a firewall and authentication/authorization
   may be performed at a proxy layer in front of the MLflow server.

2. Although artifact URLs are determined by entity metadata, the file exchange between client and cloud storage
   is always direct (i.e.: no artifacts proxy). This allows more efficient data transfer tools to be employed (e.g.: Rclone).
   Another benefit is that, as long as the entity metadata is already in the local cache, clients don't need to contact
   the MLflow server in order to fetch artifacts, which prevents the MLflow server from becoming a production dependency.

## Artifacts Lifecycle
[<img src="/docs/images/artifacts_lifecycle.jpg" height="300">]()

The dotted lines in the diagram above represent operations that are unavailable when the API is
configured in offline mode (i.e.: cache-only mode).

Artifacts can be loaded directly from the local cache or copied/symlinked to an arbitrary path.
In the case of a cache miss (or if the MLflow API is configured with `always_pull_artifacts=True`),
the artifact is automatically pulled from the cloud storage to the local cache.
All cloud to/from local file transfers are handled by the [Rclone CLI](https://rclone.org/).

Publishing an artifact file or directory causes it to be uploaded to the cloud storage at a URL determined
by the experiment run metadata(*). Optionally, a copy of the artifact is also kept in the local cache.
In order to avoid local data duplication, users may choose to have the original file or directory replaced
with as a symbolic link to the newly cached artifact.

- (*): In MLOpus, every model version must have a parent run, so the URL to a model version artifact
       is also determined through the run metadata.

### Artifact Verbs
In the MLOpus MLflow API, every method that manipulates artifacts is prefixed by one of the following verbs:

- `log_`: Upload artifact to cloud storage, optionally caching it too. Fails on offline mode.
- `pull_`: Pull artifacts from cloud storage to local cache. Fails on offline mode.
- `get_`: Get path to cached artifact. Triggers a `pull` on a cache miss or if `always_pull_artifacts=True`.
- `load_`: Deserialize cached artifact using an arbitrary loader function. Triggers a `_get`.
- `place_`: Copy or link artifact from cache to an arbitrary target path. Triggers a `_get`.
- `export_`: Same as `place_` but preserves cache folder structure, so the target path can be used as cache by another application
             (useful for building serving apps with statically linked models).

## MLOpus Kedro flavor
[<img src="/docs/images/kedro_flavor.jpg" height="600">]()

MLOpus offers some tools to be used with Kedro, which can all be employed independently but work best when used together.

The [Example 2](../examples/2_a_kedro_project) in this repository contains a simple Kedro project that showcases the usage
of the tools described here.

### Hooks

- **MlflowTracker:** Records logs, metrics, configuration, runtime params and much more in the MLflow run, in a report file
  and/or in the run metadata. Highly customizable regarding what information to record, how to filter it and how to store.

- **MlflowArtifacts:** Puts input artifacts/models in place before the pipeline runs,
  then publishes output artifacts/models on pipeline end. Validation with artifact schemas is optional.
  Setting lineage tags in the experiment run is also optional (i.e.: info regarding used inputs/outputs).

### Datasets

- **ArtifactSchemaDataset:** Saves/loads data to/from specified path using an artifact schema.
  The schema can be specified explicitly or inferred by alias from the tags of the experiment/run/model/version.

### Session

- **MlopusKedroSession:** Allows hooks and pipelines to be built dynamically by parsing the Kedro config
  (especially convenient if using Pydantic classes for nodes/hooks, so their param schema is validated
  before the pipeline starts). Also exposes session information and environment variables as config resolvers.

### Config

- **MlopusKedroConfig:** Allows applying runtime params (i.e.: config overrides) to any config scope
  (e.g.: globals, catalog, etc)

### CLI tools

The tools in `mlopus.kedro.cli_tools` allow users to define their own version of the `kedro run` command,
which can be used standalone in a custom CLI or included in the Kedro CLI itself.

Furthermore, new CLI options and callbacks can be added to the run command in order to produce side effects and/or
dynamic config overrides for each pipeline (e.g.: choosing the input model version from MLflow based on a business logic).
