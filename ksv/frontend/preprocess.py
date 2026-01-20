CUDA_KEYWORDS = [
    "__global__",
    "__device__",
    "__host__",
    "__shared__",
    "__align__",
    "__launch_bounds__",
    "__restrict__",
]

MLU_KEYWORDS = [
    "__mlu_global__",
    "__mlu_device__",
]


CUDA_BUILTINS = {
    "threadIdx.x": "threadIdx_x",
    "threadIdx.y": "threadIdx_y",
    "threadIdx.z": "threadIdx_z",
    "blockIdx.x": "blockIdx_x",
    "blockIdx.y": "blockIdx_y",
    "blockIdx.z": "blockIdx_z",
    "blockDim.x": "blockDim_x",
    "blockDim.y": "blockDim_y",
    "blockDim.z": "blockDim_z",
    "gridDim.x": "gridDim_x",
    "gridDim.y": "gridDim_y",
    "gridDim.z": "gridDim_z",
}


BUILTIN_DECLS = """
typedef struct { int x, y, z; } dim3;
dim3 threadIdx;
dim3 blockIdx;
dim3 blockDim;
dim3 gridDim;
"""


def preprocess_code(code: str, target: str) -> str:
    """
    Preprocess kernel code into C-like code suitable for AST analysis.

    Steps:
      1. Remove target-specific qualifiers
      2. Rewrite target builtins to plain identifiers
      3. Inject fake declarations for target builtins

    Args:
        code: Original kernel code
        target: Target backend ("cuda", "mlu", "cpu" or "hip")
    """

    # --------------------------------------------------
    # Step 1: Strip target keywords
    # --------------------------------------------------
    if target in ["cuda", "hip"]:
        for kw in CUDA_KEYWORDS:
            code = code.replace(kw, "")
    elif target == "mlu":
        for kw in MLU_KEYWORDS:
            code = code.replace(kw, "")

    # --------------------------------------------------
    # Step 2: Rewrite CUDA builtins
    # --------------------------------------------------
    for cuda_expr, plain_expr in CUDA_BUILTINS.items():
        code = code.replace(cuda_expr, plain_expr)

    # --------------------------------------------------
    # Step 3: Remove includes that pycparser can't handle
    # --------------------------------------------------
    code = _remove_includes(code)

    # --------------------------------------------------
    # Step 4: Inject fake builtin declarations
    # --------------------------------------------------
    # code = BUILTIN_DECLS + "\n" + code

    return code


def _remove_includes(code: str) -> str:
    """
    Remove #include lines (pycparser doesn't need them).
    """
    lines = []
    for line in code.splitlines():
        if line.strip().startswith("#include"):
            continue
        lines.append(line)
    return "\n".join(lines)
