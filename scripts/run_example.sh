#!/usr/bin/env zsh
# Simple helper to run the demo validation for the example kernel.
set -euo pipefail
python3 examples/run_ksv.py examples/kernels/vec_add.cu
python3 examples/run_ksv.py examples/kernels/vec_max.cu
