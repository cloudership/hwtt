#!/bin/bash

set -euo pipefail

ROOT="$(dirname "${BASH_SOURCE[0]}")"

. "${ROOT}/.env"

mlflow ui --port 8080 --backend-store-uri "${MLFLOW_TRACKING_URI}"
