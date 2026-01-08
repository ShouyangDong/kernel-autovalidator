#!/usr/bin/env python3
"""Minimal end-to-end driver for Kernel-AutoValidator (demo implementation).

Usage:
  python validator/validate.py examples/vec_add.cu

This script runs a lightweight pipeline:
  parser -> analysis -> codegen (harness scaffold) -> runtime (simple simulation)

The implementation is intentionally small and safe to run without CUDA.
"""

import json
import os
import sys

from validator.parser import parser as parser_mod
from validator.analysis import infer as infer_mod
from validator.codegen import generate as gen_mod
from validator.runtime import checks as checks_mod


def validate(kernel_path: str) -> None:
    parsed = parser_mod.parse_kernel(kernel_path)
    print('Parsed kernel:')
    print('  name:', parsed['name'])
    print('  params:', parsed['params'])
    print('  index_expr:', parsed.get('index_expr'))

    analysis = infer_mod.infer(parsed)
    print('\nInferred roles:')
    for p, r in analysis['roles'].items():
        print(f'  {p}: {r}')
    print('Execution config:', analysis['exec_config'])

    # write parsed and analysis JSON for downstream tools
    base = os.path.splitext(os.path.basename(kernel_path))[0]
    parsed_json = base + '.parsed.json'
    analysis_json = base + '.analysis.json'
    with open(parsed_json, 'w', encoding='utf-8') as f:
        json.dump(parsed, f, indent=2)
    with open(analysis_json, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2)

    harness_path = base + '.harness.txt'
    summary_path = gen_mod.generate_harness(parsed, analysis, harness_path)
    print('\nWrote harness scaffold to', harness_path)
    print('Wrote summary to', summary_path)

    # Run the simple runtime checks/simulation
    report = checks_mod.simulate_kernel(parsed, analysis)
    print('\nRuntime simulation report:')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python validate.py <kernel.cu>')
        raise SystemExit(1)
    validate(sys.argv[1])
