# ksv/pipeline.py

from typing import Dict, Tuple

from ksv.analysis.bound_infer import infer_tensor_shapes
from ksv.analysis.buffer_roles import infer_buffer_roles
from ksv.analysis.index_expr import extract_index_expressions
from ksv.analysis.loop_bound import extract_loop_bounds
from ksv.analysis.memory_access import extract_memory_accesses
from ksv.execution.config_infer import infer_execution_config
from ksv.execution.host_codegen import generate_host_code
from ksv.frontend.ast_builder import build_ast
from ksv.frontend.preprocess import preprocess_code
from ksv.validation.runner import run_kernel


class KSVExecutionResult:
    def __init__(
        self,
        outputs: Dict[str, object],
        tensor_shapes: Dict[str, Tuple[int]],
        exec_config: Dict,
    ):
        self.outputs = outputs
        self.tensor_shapes = tensor_shapes
        self.exec_config = exec_config

    def __repr__(self):
        return (
            f"KSVExecutionResult("
            f"outputs={list(self.outputs.keys())}, "
            f"tensor_shapes={self.tensor_shapes}, "
            f"exec_config={self.exec_config})"
        )


def run_ksv(
    kernel_path: str,
    target: str = "cuda",
    verbose: bool = False,
) -> KSVExecutionResult:
    """
    Kernel Semantic Validation (KSV) execution pipeline.

    This pipeline:
      1) infers buffer roles
      2) infers safe tensor shapes
      3) infers execution configuration
      4) generates host code
      5) executes the kernel and returns outputs

    Args:
        kernel_path: CUDA / HIP kernel source
        target: Target backend ("cuda", "mlu", "cpu" or "hip")

    Returns:
        KSVExecutionResult
    """
    if verbose:
        print(f"[KSV] Validating kernel: {kernel_path}")
    # --------------------------------------------------
    # Step 0: Load & preprocess kernel
    # --------------------------------------------------
    with open(kernel_path, "r") as f:
        kernel_code = preprocess_code(f.read(), target=target)

    # --------------------------------------------------
    # Step 1: AST construction
    # --------------------------------------------------
    kernel_ast = build_ast(kernel_code)

    # --------------------------------------------------
    # Step 2: Buffer role inference
    # --------------------------------------------------
    buffer_roles = infer_buffer_roles(kernel_ast)
    # { "input": [...], "output": [...], "inout": [...] }
    # --------------------------------------------------
    # Step 3: Memory access & index analysis
    # --------------------------------------------------
    accesses = extract_memory_accesses(kernel_ast, buffer_roles)
    extract_index_expressions(accesses)
    # --------------------------------------------------
    # Step 4: Loop / guard bound extraction
    # --------------------------------------------------
    loop_bounds = extract_loop_bounds(kernel_ast)
    # --------------------------------------------------
    # Step 5: Tensor shape inference
    # --------------------------------------------------
    # Note: `infer_tensor_shapes` currently expects (memory_accesses, loop_bounds, thread_bounds)
    # We don't yet compute separate `thread_bounds` here, so pass an empty
    # dict for now.
    tensor_shapes = infer_tensor_shapes(
        memory_accesses=accesses,
        loop_bounds=loop_bounds,
        thread_bounds={},
    )

    # --------------------------------------------------
    # Step 6: Execution configuration inference
    # --------------------------------------------------
    exec_config = infer_execution_config(kernel_ast, loop_bounds)

    # --------------------------------------------------
    # Step 7: Host code generation + execution
    # --------------------------------------------------
    # with tempfile.TemporaryDirectory() as tmpdir:
    # Use actual kernel name from AST when generating host code
    kernel_name = getattr(kernel_ast.decl, "name", "kernel")
    host_path = generate_host_code(
        kernel_code=kernel_code,
        kernel_name=kernel_name,
        buffer_roles=buffer_roles,
        tensor_shapes=tensor_shapes,
        exec_config=exec_config,
        out_dir="./",
        kernel_ast=kernel_ast,
        target=target,
    )

    # Also generate a Python ctypes runner that will load the compiled library
    # (expected to be produced by building `host.cu` into a shared library).
    try:
        from ksv.execution.python_runner_codegen import generate_python_runner

        # infer element counts and param types
        # buffer_roles is dict name->role
        buffers = buffer_roles
        elem_counts = {}
        for name in buffers:
            shape = tensor_shapes.get(name, (1024,))
            # flatten shape to element count
            if isinstance(shape, int):
                cnt = shape
            else:
                cnt = 1
                for s in shape:
                    cnt *= s
            elem_counts[name] = cnt

        # extract param_types from kernel_ast if available
        param_types = {}
        if kernel_ast is not None:
            try:
                func_decl = kernel_ast.decl.type
                if getattr(func_decl, "args", None) is not None:
                    for param in func_decl.args.params:
                        name = getattr(param, "name", None)
                        ty = getattr(param, "type", None)
                        if (
                            ty is not None
                            and ty.__class__.__name__ == "PtrDecl"
                        ):
                            inner = ty.type
                            if inner.__class__.__name__ == "TypeDecl":
                                it = inner.type
                                if it.__class__.__name__ == "IdentifierType":
                                    param_types[name] = " ".join(it.names)
            except Exception:
                param_types = {}

        # pick library name (assume host.cu compiled as libhost.so or
        # libhost.dylib)
        lib_name = "./libhost.so"
        py_runner = generate_python_runner(
            out_dir=".",
            lib_name=lib_name,
            buffers=buffers,
            elem_counts=elem_counts,
            param_types=param_types,
        )
    except Exception:
        pass

    outputs = run_kernel(host_path)

    # --------------------------------------------------
    # Done
    # --------------------------------------------------
    return KSVExecutionResult(
        outputs=outputs,
        tensor_shapes=tensor_shapes,
        exec_config=exec_config,
    )
