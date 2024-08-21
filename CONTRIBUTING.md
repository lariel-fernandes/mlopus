# Contributing

## Issues

### Before opening an issue
1. [Search the open and closed issues](https://github.com/lariel-fernandes/mlopus/issues?q=is%3Aissue) to see if it already exists.
2. Have a look at the [architecture guide](docs/architecture.md) to familiarize yourself with this project's principles, intended usage, base assumptions and by-design limitations.

### Opening an `enhancement` issue
1. Use the `enhancement` tag
2. Explain the use case or motivation
3. If relevant, provide a snippet (possibly in meta-code) that demonstrates the intended usage of the feature

### Opening a `bug` issue
1. Use the tag `bug`
2. Indicate the affected versions
3. Provide:
   - Steps to reproduce
   - Your environment's details
   - A file with the complete logs/outputs/stacktrace in case of an exception(*)

(*): You may want to redact any personal/private details or confidential info

## Opening a Pull Request
1. First, pick an open issue or [open a new one](#before-opening-an-issue).
2. Follow the [Developer Instructions](#developer-instructions) below to [install the recommended software](#recommended-software) and [set up your environment](#environment-setup)
3. Make your changes and validate them with `make test`
4. Push your changes and open the PR:
   - The PR should include a link to the issue
   - The PR name should comply with the [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/#examples) pattern

## Developer instructions

### Recommended Software
1. [UV](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)
2. [Rclone CLI](https://rclone.org/install/#script-installation) (required for artifact transfer from/to cloud storage)

### Environment setup
1. Fork and clone this repo

2. From the cloned repo's root, run `make install`
   This is going to:
   - Install the project source code and requirements to a virtualenv
   - Install the pre-commit hooks

3. The Python interpreter path for your IDE can be obtained with `uv run which python`
