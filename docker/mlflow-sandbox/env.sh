#!/bin/bash

script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

source "${script_dir}/.env"

export RCLONE_CONFIG_S3_TYPE=s3
export RCLONE_CONFIG_S3_ENV_AUTH=true
export RCLONE_CONFIG_S3_PROVIDER=Other
export RCLONE_CONFIG_S3_DECOMPRESS=true
export RCLONE_CONFIG_S3_ACCESS_KEY_ID=${LOCALSTACK_ACCESS_KEY_ID}
export RCLONE_CONFIG_S3_SECRET_ACCESS_KEY=${LOCALSTACK_SECRET_ACCESS_KEY}
export RCLONE_CONFIG_S3_ENDPOINT=http://localhost:${LOCALSTACK_HOST_PORT}

export MLFLOW_TRACKING_URI=http://localhost:${MLFLOW_HOST_PORT}
