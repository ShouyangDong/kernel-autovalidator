import textwrap
from typing import Dict


def generate_python_runner(
    out_dir: str,
    lib_name: str,
    buffers: Dict[str, str],
    elem_counts: Dict[str, int],
    param_types: Dict[str, str],
):
    """Emit a Python ctypes runner that loads `lib_name` and calls `run_kernel`.

    Args:
      out_dir: output directory where runner will be written
      lib_name: path to shared library (.so/.dylib)
      buffers: ordered dict mapping buffer name -> role (input/output/inout)
      elem_counts: mapping buffer -> element count (int)
      param_types: mapping buffer -> C base type (e.g., 'float')
    """
    # Build imports and numpy allocations
    lines = [
        "import ctypes",
        "import numpy as np",
        "import os",
        "",
    ]

    lines.append(f"lib = ctypes.CDLL('{lib_name}')")
    lines.append("")

    # build argtypes for run_kernel
    argtypes = []
    for buf in buffers:
        # map C base type to numpy dtype and ctypes pointer
        base = param_types.get(buf, "float")
        if base == "float":
            ctype = "ctypes.POINTER(ctypes.c_float)"
        elif base == "double":
            ctype = "ctypes.POINTER(ctypes.c_double)"
        elif base in ("int", "unsigned int"):
            ctype = "ctypes.POINTER(ctypes.c_int)"
        else:
            ctype = "ctypes.POINTER(ctypes.c_float)"
        argtypes.append(ctype)
    # set argtypes on lib.run_kernel
    lines.append("# Configure run_kernel signature")
    lines.append("lib.run_kernel.argtypes = [")
    for a in argtypes:
        lines.append(f"    {a},")
    lines.append("]")
    lines.append("")

    # allocate host buffers
    lines.append("# Allocate host buffers (NumPy)")
    for buf in buffers:
        count = elem_counts.get(buf, 1024)
        base = param_types.get(buf, "float")
        if base == "float":
            np_type = "np.float32"
        elif base == "double":
            np_type = "np.float64"
        elif base in ("int", "unsigned int"):
            np_type = "np.int32"
        else:
            np_type = "np.float32"
        lines.append(f"h_{buf} = np.zeros({count}, dtype={np_type})")
        if buffers[buf] in ("input", "inout"):
            lines.append(
                f"h_{buf}[:] = np.arange({count}, dtype={np_type}) % 13"
            )
    lines.append("")

    # call run_kernel
    call_args = ", ".join(
        [
            f"h_{b}.ctypes.data_as(ctypes.POINTER(ctypes.c_float))"
            for b in buffers
        ]
    )
    lines.append("# Call the run_kernel function")
    lines.append(f"lib.run_kernel({call_args})")
    lines.append("")

    content = "\n".join(lines)

    path = out_dir.rstrip("/") + "/run_kernel.py"
    with open(path, "w") as f:
        f.write(textwrap.dedent(content))

    return path
