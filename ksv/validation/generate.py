import json
import os
import re
from typing import Dict


def generate_harness(parsed: Dict, analysis: Dict, out_path: str) -> str:
    """Generate CUDA source, a build script, and a Python ctypes runner.

    - `parsed` is the output of `parser.parse_kernel`.
    - `analysis` is the output of `analysis.infer`.
    - `out_path` is the desired harness filename (used to derive base names).

    The function writes:
      - `<base>.lib.cu` : CUDA source with device kernel + host wrapper `run_<name>`
      - `<base>.build.sh`: script that invokes `nvcc` to build a shared lib
      - `<base>.runner.py`: Python script that builds (if needed), calls via ctypes,
                           and does a simple golden check for elementwise add.
      - `<out_path>.summary.json`: metadata describing generated files

    This generator assumes pointer parameters are `float*` and a single
    1-D tensor of length `N` inferred by analysis; it's intended as a
    small, inspectable demo harness.
    """

    name = parsed["name"]
    params = parsed["params"]
    roles = analysis["roles"]
    cfg = analysis["exec_config"]

    block_dim = cfg.get("blockDim", 256)
    N = cfg.get("N", 1024)

    # Use parsed param types when available
    param_types = parsed.get("param_types", {})

    def c_type_for(p):
        info = param_types.get(p, None)
        if info is None:
            return "float*"
        base = info.get("base", "float").strip()
        is_ptr = info.get("is_pointer", True)
        return f"{base}{'*' if is_ptr else ''}"

    # Build parameter lists
    param_list = ", ".join([f"{c_type_for(p)} {p}" for p in params])

    # Prepare device variable names and helper lines
    dev_names = [f"{p}_d" for p in params]
    alloc_lines = []
    copy_to_device_lines = []
    copy_to_host_lines = []
    free_lines = []
    # Ensure reads/writes sets only include declared params
    parsed_reads = set(parsed.get("reads", [])) & set(params)
    parsed_writes = set(parsed.get("writes", [])) & set(params)

    for p, d in zip(params, dev_names):
        info = param_types.get(p, {})
        is_ptr = info.get("is_pointer", True)
        role = roles.get(p, None)
        if is_ptr:
            alloc_lines.append(f"  {c_type_for(p)} {d} = nullptr;")
            alloc_lines.append(
                f"  cudaError_t err_{d} = cudaMalloc((void**)&{d}, bytes);"
            )
            alloc_lines.append(
                f'  if (err_{d} != cudaSuccess) {{ fprintf(stderr, "cudaMalloc {p} failed\\n"); return; }}'
            )
            # Host -> Device copies: only for inputs or in-place (or if parser
            # marked as read)
            if role in ("input", "in-place") or p in parsed_reads:
                copy_to_device_lines.append(
                    f"  cudaMemcpy({d}, {p}_h, bytes, cudaMemcpyHostToDevice);"
                )
            # Device -> Host copies: only for outputs or in-place (or if parser
            # marked as write)
            if role in ("output", "in-place") or p in parsed_writes:
                copy_to_host_lines.append(
                    f"  cudaMemcpy({p}_h, {d}, bytes, cudaMemcpyDeviceToHost);"
                )
            free_lines.append(f"  cudaFree({d});")
        else:
            # scalar: no device allocation or copy
            alloc_lines.append(f"  /* scalar {p} (no device allocation) */")

    # Prepare wrapper parameter list (host-side args)
    wrapper_params = ", ".join(
        [f"{c_type_for(p)} {p}_h" for p in params] + ["int N"]
    )

    # Construct CUDA source
    kernel_src = f"""#include <cuda_runtime.h>
#include <stdio.h>

__global__ void {name}({param_list}) {{
{parsed['body']}
}}

extern "C" void run_{name}({wrapper_params}) {{
  size_t bytes = (size_t)N * sizeof(float);
{os.linesep.join(alloc_lines)}

{os.linesep.join(copy_to_device_lines)}

  dim3 block({block_dim});
  dim3 grid((N + block.x - 1) / block.x);
  {name}<<<grid, block>>>({', '.join([d if param_types.get(p, {}).get('is_pointer', True) else p + '_h' for p, d in zip(params, dev_names)])});

{os.linesep.join(copy_to_host_lines)}

{os.linesep.join(free_lines)}
}}
"""

    base = os.path.splitext(out_path)[0]
    cu_path = base + ".lib.cu"
    with open(cu_path, "w", encoding="utf-8") as f:
        f.write(kernel_src)

    # Build script
    lib_name = f"lib{os.path.basename(base)}.so"
    build_sh = f"""#!/usr/bin/env bash
set -euo pipefail
nvcc -shared -Xcompiler -fPIC -o {lib_name} {os.path.basename(cu_path)}
"""
    build_path = base + ".build.sh"
    with open(build_path, "w", encoding="utf-8") as f:
        f.write(build_sh)

    # Detect simple elementwise add pattern for golden computation
    add_match = re.search(
        r"(\w+)\s*\[\s*[^\]]+\s*\]\s*=\s*(\w+)\s*\[\s*[^\]]+\s*\]\s*\+\s*(\w+)\s*\[",
        parsed["body"],
    )

    # Build Python runner
    runner_lines = [
        "#!/usr/bin/env python3",
        "import ctypes",
        "import numpy as np",
        "import subprocess, os, sys",
        "",
        f"LIB = './{lib_name}'",
        "",
        "def build():",
        f"    subprocess.check_call(['chmod', '+x', '{
            os.path.basename(build_path)}'])",
        f"    subprocess.check_call(['./{os.path.basename(build_path)}'])",
        "",
        "def run(N=None):",
        f"    if N is None: N = {N}",
    ]

    # Create host arrays / scalars per role and param type
    for p in params:
        info = param_types.get(p, {})
        is_ptr = info.get("is_pointer", True)
        base = info.get("base", "float").strip()
        role = roles.get(p, "unknown")
        if is_ptr:
            # pointer -> host buffer
            if "float" in base:
                if role == "output":
                    runner_lines.append(
                        f"    {p} = np.zeros(N, dtype=np.float32)"
                    )
                else:
                    runner_lines.append(
                        f"    {p} = np.random.rand(N).astype(np.float32)"
                    )
            else:
                # default to float buffers for unknown base
                runner_lines.append(
                    f"    {p} = np.random.rand(N).astype(np.float32)"
                )
        else:
            # scalar parameter
            if base in ("int", "unsigned int", "long", "size_t"):
                # default scalar sizes to N when appropriate
                runner_lines.append(f"    {p} = int({N})")
            elif "float" in base:
                runner_lines.append(f"    {p} = float({N})")
            else:
                runner_lines.append(f"    {p} = 0  # scalar default")

    runner_lines += [
        "",
        "    if not os.path.exists(LIB):",
        "        print('Building library...')",
        "        build()",
        "",
        "    lib = ctypes.CDLL(LIB)",
    ]

    # argtypes: build ctypes signature based on param types
    argtypes = []
    for p in params:
        info = param_types.get(p, {})
        is_ptr = info.get("is_pointer", True)
        base = info.get("base", "float").strip()
        if is_ptr:
            # assume float pointers for ctypes
            argtypes.append("ctypes.POINTER(ctypes.c_float)")
        else:
            if base in ("int", "unsigned int", "long", "size_t"):
                argtypes.append("ctypes.c_int")
            elif "float" in base:
                argtypes.append("ctypes.c_float")
            else:
                argtypes.append("ctypes.c_int")
    argtypes.append("ctypes.c_int")  # final N
    runner_lines.append(f"    run = lib.run_{name}")
    runner_lines.append(f"    run.argtypes = [{', '.join(argtypes)}]")
    runner_lines.append("    run.restype = None")
    runner_lines.append("")

    # Prepare arguments and call
    call_items = []
    for p in params:
        info = param_types.get(p, {})
        is_ptr = info.get("is_pointer", True)
        base = info.get("base", "float").strip()
        if is_ptr:
            runner_lines.append(
                f"    {p}_ptr = {p}.ctypes.data_as(ctypes.POINTER(ctypes.c_float))"
            )
            call_items.append(f"{p}_ptr")
        else:
            # scalar: pass as ctypes scalar
            if base in ("int", "unsigned int", "long", "size_t"):
                call_items.append(f"ctypes.c_int({p})")
            elif "float" in base:
                call_items.append(f"ctypes.c_float({p})")
            else:
                call_items.append(f"ctypes.c_int({p})")

    call_items.append("ctypes.c_int(N)")
    call_args = ", ".join(call_items)
    runner_lines.append(f"    run({call_args})")
    runner_lines.append("")

    # Golden check for simple add
    if add_match:
        out_name = add_match.group(1)
        a_name = add_match.group(2)
        b_name = add_match.group(3)
        runner_lines += [
            "    # Simple golden check for elementwise add",
            f"    gold = {a_name} + {b_name}",
            f"    if not np.allclose({out_name}, gold, atol=1e-6):",
            "        print('Validation failed: output does not match golden')",
            "        sys.exit(1)",
            "    print('Validation passed: output matches golden')",
        ]
    else:
        runner_lines.append(
            "    print('No simple golden available; inspect outputs manually.')"
        )

    runner_lines += ["", "if __name__ == '__main__':", "    run()"]

    runner_path = base + ".runner.py"
    with open(runner_path, "w", encoding="utf-8") as f:
        f.write("\n".join(runner_lines))

    summary = {
        "kernel": name,
        "params": params,
        "roles": roles,
        "exec_config": cfg,
        "cuda_source": cu_path,
        "build_script": build_path,
        "shared_lib": lib_name,
        "runner": runner_path,
    }
    summary_path = out_path + ".summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print(
            "Usage: python generate.py <parsed.json> <analysis.json> <out_harness.txt>"
        )
        raise SystemExit(1)
    parsed = json.load(open(sys.argv[1], "r"))
    analysis = json.load(open(sys.argv[2], "r"))
    print("Wrote summary to", generate_harness(parsed, analysis, sys.argv[3]))
