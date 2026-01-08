import json
from typing import Dict


def generate_harness(parsed: Dict, analysis: Dict, out_path: str) -> str:
    """Generate a minimal host-side harness stub (text) and write a JSON summary.

    The harness is a textual scaffold showing how to allocate buffers and
    launch the kernel. We don't compile it here; this file is intended to be
    small and easy to inspect.
    """
    name = parsed['name']
    params = parsed['params']
    roles = analysis['roles']
    cfg = analysis['exec_config']

    harness = []
    harness.append(f"// Harness for kernel: {name}")
    harness.append(f"// Inferred N = {cfg['N']}, blockDim = {cfg['blockDim']}, gridDim = {cfg['gridDim']}")
    harness.append('')
    harness.append('// Buffer roles:')
    for p in params:
        harness.append(f"//   {p}: {roles.get(p, 'unknown')}")

    harness.append('')
    harness.append('// Example host code (pseudo):')
    harness.append('')
    harness.append('/*')
    harness.append('float *A = malloc(N * sizeof(float));')
    harness.append('float *B = malloc(N * sizeof(float));')
    harness.append('float *C = malloc(N * sizeof(float));')
    harness.append('init(A); init(B);')
    harness.append('vec_add<<<gridDim, blockDim>>>(A, B, C);')
    harness.append('cudaDeviceSynchronize();')
    harness.append('validate(C);')
    harness.append('*/')

    harness_text = '\n'.join(harness)

    # write harness file and summary
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(harness_text)

    summary = {
        'kernel': name,
        'params': params,
        'roles': roles,
        'exec_config': cfg,
        'harness_file': out_path,
    }
    summary_path = out_path + '.summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    return summary_path


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 4:
        print('Usage: python generate.py <parsed.json> <analysis.json> <out_harness.txt>')
        raise SystemExit(1)
    import json
    parsed = json.load(open(sys.argv[1], 'r'))
    analysis = json.load(open(sys.argv[2], 'r'))
    print('Wrote summary to', generate_harness(parsed, analysis, sys.argv[3]))
