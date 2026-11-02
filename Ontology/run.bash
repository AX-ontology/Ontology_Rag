#!/usr/bin/env bash
set -euo pipefail
ONTOLOGY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ONTOLOGY_DIR/final_pipeline/run_all.bash" "$@"
