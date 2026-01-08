import math
from typing import Dict


def infer(parsed: Dict) -> Dict:
    """Infer buffer roles, simple shapes, and execution configuration.

    This is intentionally conservative and minimal: it uses parsed read/write
    sets and common index-expression patterns to derive a safe tensor size
    and an example grid/block configuration.
    """
    params = parsed.get('params', [])
    reads = set(parsed.get('reads', []))
    writes = set(parsed.get('writes', []))

    roles = {}
    for p in params:
        if p in reads and p in writes:
            roles[p] = 'in-place'
        elif p in writes:
            roles[p] = 'output'
        elif p in reads:
            roles[p] = 'input'
        else:
            roles[p] = 'unknown'

    # Infer a safe tensor shape N from index expression, fallback to 1024
    index_expr = parsed.get('index_expr') or ''
    if 'gridDim' in index_expr or 'blockDim' in index_expr:
        # default blockDim = 256
        block_dim = 256
        grid_dim = 4
        N = block_dim * grid_dim
    else:
        N = 1024
        block_dim = 256
        grid_dim = math.ceil(N / block_dim)

    exec_config = {
        'blockDim': block_dim,
        'gridDim': grid_dim,
        'N': N,
    }

    return {
        'roles': roles,
        'exec_config': exec_config,
        'index_expr': index_expr,
    }


if __name__ == '__main__':
    import sys, json
    if len(sys.argv) < 2:
        print('Usage: python infer.py <parsed.json>')
        raise SystemExit(1)
    parsed = json.load(open(sys.argv[1], 'r'))
    print(json.dumps(infer(parsed), indent=2))
