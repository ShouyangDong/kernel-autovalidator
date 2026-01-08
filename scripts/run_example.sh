#!/usr/bin/env zsh
# Simple helper to run the demo validation for the example kernel.
set -euo pipefail
python3 validator/validate.py examples/vec_add.cu
