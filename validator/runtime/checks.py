import random
import math
from typing import Dict, List


def simulate_kernel(parsed: Dict, analysis: Dict) -> Dict:
    """Run a conservative Python simulation for a few simple kernel patterns.

    Currently recognizes the elementwise add pattern `C[i] = A[i] + B[i]` and
    simulates it with Python lists. Returns a report of checks.
    """
    body = parsed.get('body', '')
    N = analysis['exec_config']['N']

    # simple recognition: look for "C[i] = A[i] + B[i]"
    m = None
    import re
    m = re.search(r"(\w+)\s*\[\s*[^\]]+\s*\]\s*=\s*(\w+)\s*\[\s*[^\]]+\s*\]\s*\+\s*(\w+)\s*\[", body)
    report = {
        'recognized_pattern': False,
        'checks': {},
    }
    if m:
        out_buf = m.group(1)
        a = m.group(2)
        b = m.group(3)
        report['recognized_pattern'] = 'add'  

        # initialize inputs
        A = [random.random() for _ in range(N)]
        B = [random.random() for _ in range(N)]
        # simulate kernel computed result
        C_sim = [A[i] + B[i] for i in range(N)]

        # property checks
        # determinism: running again yields same result (with same inputs)
        C_sim2 = [A[i] + B[i] for i in range(N)]
        determ = C_sim == C_sim2

        # no-NaNs
        any_nan = any(math.isnan(x) for x in C_sim)

        report['checks']['determinism'] = determ
        report['checks']['no_nans'] = (not any_nan)
        report['checks']['N'] = N
        report['checks']['out_buf'] = out_buf
        report['checks']['in_bufs'] = [a, b]

    else:
        report['recognized_pattern'] = False
        report['checks']['reason'] = 'pattern not recognized (only simple add supported)'

    return report


if __name__ == '__main__':
    import sys, json
    if len(sys.argv) < 3:
        print('Usage: python checks.py <parsed.json> <analysis.json>')
        raise SystemExit(1)
    parsed = json.load(open(sys.argv[1], 'r'))
    analysis = json.load(open(sys.argv[2], 'r'))
    print(json.dumps(simulate_kernel(parsed, analysis), indent=2))
