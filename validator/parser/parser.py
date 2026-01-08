import re
from typing import Dict, List, Set


def parse_kernel(file_path: str) -> Dict:
    """Parse a simple CUDA kernel file and extract kernel name, params,
    body, simple read/write sets, and an index expression if present.

    This is a lightweight, best-effort parser intended for small examples
    (it uses regexes)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        src = f.read()

    m = re.search(r"__global__\s+void\s+(\w+)\s*\(([^)]*)\)\s*\{(.*)\}", src, re.S)
    if not m:
        raise ValueError(f"No __global__ kernel found in {file_path}")

    name = m.group(1)
    raw_params = m.group(2).strip()
    body = m.group(3).strip()

    params = []
    if raw_params:
        for p in raw_params.split(','):
            p = p.strip()
            # match patterns like "float* A" or "const float* X"
            pm = re.search(r"(\w+)\s*(?:\*\s*)?(\w+)$", p)
            if pm:
                params.append(pm.group(2))
            else:
                # fallback: take last token
                params.append(p.split()[-1])

    # Find simple index expression (common pattern)
    idx_expr = None
    idx_match = re.search(r"int\s+\w+\s*=\s*([^;]+);", body)
    if idx_match:
        expr = idx_match.group(1).strip()
        if any(s in expr for s in ("blockIdx", "blockDim", "threadIdx")):
            idx_expr = expr

    # Find write targets (LHS of assignment)
    writes: Set[str] = set()
    reads: Set[str] = set()

    # Find all array accesses like A[i]
    array_refs = re.findall(r"(\w+)\s*\[\s*([^\]]+)\s*\]", body)
    # Determine whether each access is on LHS or RHS
    for ref in array_refs:
        name_ref, idx = ref
        # naive: if the pattern "name[idx] =" appears, it's a write
        if re.search(rf"{re.escape(name_ref)}\s*\[\s*{re.escape(idx)}\s*\]\s*=", body):
            writes.add(name_ref)
        else:
            reads.add(name_ref)

    # Also collect scalar vars used on RHS (very small heuristic)
    rhs_vars = re.findall(r"=([^;]+);", body)
    for chunk in rhs_vars:
        for p in re.findall(r"\b(\w+)\b", chunk):
            if p in params:
                reads.add(p)

    return {
        'file': file_path,
        'name': name,
        'params': params,
        'body': body,
        'index_expr': idx_expr,
        'reads': sorted(reads),
        'writes': sorted(writes),
    }


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python parser.py <kernel.cu>')
        raise SystemExit(1)
    info = parse_kernel(sys.argv[1])
    import json
    print(json.dumps(info, indent=2))
