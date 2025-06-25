#!/bin/bash

set -euo pipefail

ROOT="$(dirname "${BASH_SOURCE[0]}")"

. "${ROOT}/.env"

mlflow server --host 0.0.0.0 --port "${MLFLOW_TRACKING_PORT}" --backend-store-uri "${BACKEND_STORE_URI}" --artifacts-destination "${MLFLOW_ARTIFACTS}"
