import os
import textwrap


def num_elements(shape):
    # Accept either an int (total elements) or an iterable shape tuple
    if isinstance(shape, int):
        return shape
    n = 1
    for s in shape:
        n *= s
    return n


headers = {
    "cuda": "#include <cuda.h>\n#include <cuda_runtime.h>",
    "mlu": "#include <bang.h>\n",
    "hip": "#include <hip/hip_runtime.h>",
    "cpu": "#include <stdio.h>",
}


def generate_host_code(
    kernel_code,
    kernel_name,
    buffer_roles,
    tensor_shapes,
    exec_config,
    out_dir,
    kernel_ast=None,
    target="cuda",
):
    os.makedirs(out_dir, exist_ok=True)
    if target == "cuda":
        host_path = os.path.join(out_dir, "host.cu")
    elif target == "mlu":
        host_path = os.path.join(out_dir, "host.mlu")
    elif target == "hip":
        host_path = os.path.join(out_dir, "host.hip")
    else:
        raise NotImplementedError(f"Unsupported target: {target}")

    block_x, block_y, block_z = exec_config["block"]
    grid_x, grid_y, grid_z = exec_config["grid"]

    # Build lists of buffers by role (the analyser returns a dict mapping
    # name->role)
    input_bufs = [
        name for name, role in buffer_roles.items() if role == "input"
    ]
    output_bufs = [
        name for name, role in buffer_roles.items() if role == "output"
    ]
    inout_bufs = [
        name for name, role in buffer_roles.items() if role == "inout"
    ]

    # Determine C element types for buffers from kernel AST when possible
    param_types = {}
    if kernel_ast is not None:
        try:
            func_decl = kernel_ast.decl.type
            if getattr(func_decl, "args", None) is not None:
                for param in func_decl.args.params:
                    name = getattr(param, "name", None)
                    # find base type for pointer parameters
                    ty = getattr(param, "type", None)
                    base = None
                    if ty is not None:
                        # PtrDecl -> TypeDecl -> IdentifierType
                        if ty.__class__.__name__ == "PtrDecl":
                            inner = ty.type
                            if inner.__class__.__name__ == "TypeDecl":
                                it = inner.type
                                if it.__class__.__name__ == "IdentifierType":
                                    base = " ".join(it.names)
                    if name is not None and base is not None:
                        param_types[name] = base
        except Exception:
            param_types = {}

    # === Declare buffers ===
    decls = []
    mallocs = []
    h_mallocs = []
    h2d = []
    d2h = []
    frees = []

    for buf in input_bufs + output_bufs + inout_bufs:
        shape = tensor_shapes.get(buf, (1024,))
        n_elem = num_elements(shape)
        size = n_elem * 4  # float

        decls.append(f"float *d_{buf}; float *h_{buf};")

        mallocs.append(f"cudaMalloc(&d_{buf}, {size});")
        h_mallocs.append(f"h_{buf} = (float*)malloc({size});")

        if buf in input_bufs or buf in inout_bufs:
            h2d.append(
                f"cudaMemcpy(d_{buf}, h_{buf}, {size}, cudaMemcpyHostToDevice);"
            )

        if buf in output_bufs or buf in inout_bufs:
            d2h.append(
                f"cudaMemcpy(h_{buf}, d_{buf}, {size}, cudaMemcpyDeviceToHost);"
            )

        frees.append(f"cudaFree(d_{buf}); free(h_{buf});")

    # === Kernel argument list ===
    kernel_args = []
    for buf in input_bufs + output_bufs + inout_bufs:
        kernel_args.append(f"d_{buf}")
    kernel_args_str = ", ".join(kernel_args)

    # === Host code ===
    # Generate a shared-library friendly CU that exposes an extern "C" launcher
    # function callable from ctypes. The launcher will take host pointers for
    # each buffer and perform H2D -> kernel launch -> D2H. No stdout printing.

    # Build C signature for run_kernel using host buffer names and parameter
    # types
    run_args = []
    # mapping base type -> byte size
    type_size = {
        "float": 4,
        "double": 8,
        "int": 4,
        "unsigned int": 4,
        "half": 2,
    }
    elem_size_map = {}
    for buf in input_bufs + output_bufs + inout_bufs:
        base = param_types.get(buf, "float")
        ctype = base
        run_args.append(f"{ctype}* h_{buf}")
        elem_size_map[buf] = type_size.get(base, 4)

    # No runtime `n` parameter: sizes are derived from `tensor_shapes`
    # statically.
    run_args_str = ", ".join(run_args)

    # Use exec_config constants for launch configuration
    bx, by, bz = block_x, block_y, block_z
    gx, gy, gz = grid_x, grid_y, grid_z

    # Precompute per-buffer element counts and byte sizes using tensor_shapes
    elem_count_map = {}
    byte_size_map = {}
    for buf in input_bufs + output_bufs + inout_bufs:
        shape = tensor_shapes.get(buf, (1024,))
        count = num_elements(shape)
        elem_count_map[buf] = count
        byte_size_map[buf] = count * elem_size_map.get(buf, 4)

    host_code = f"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ {kernel_code}

extern "C" void run_kernel({run_args_str}) {{
    // Device pointers (use per-buffer types)
{textwrap.indent('\n'.join([f"{param_types.get(b, 'float')} *d_{b};" for b in input_bufs + output_bufs + inout_bufs]), '    ')}

    // Device malloc and host->device copies using statically-derived sizes
{textwrap.indent('\n'.join([f'cudaMalloc(&d_{b}, (size_t){byte_size_map.get(b, 4)});' for b in input_bufs + output_bufs + inout_bufs]), '    ')}
{textwrap.indent('\n'.join([f'if (h_{b}) cudaMemcpy(d_{b}, h_{b}, (size_t){byte_size_map.get(b, 4)}, cudaMemcpyHostToDevice);' for b in input_bufs + inout_bufs]), '    ')}

    // Launch kernel
    dim3 block({bx}, {by}, {bz});
    dim3 grid({gx}, {gy}, {gz});
    {kernel_name}<<<grid, block>>>({kernel_args_str});

    // Device -> host copies for outputs
{textwrap.indent('\n'.join([f'if (h_{b}) cudaMemcpy(h_{b}, d_{b}, (size_t){byte_size_map.get(b, 4)}, cudaMemcpyDeviceToHost);' for b in output_bufs + inout_bufs]), '    ')}

    // Free device memory
{textwrap.indent('\n'.join([f'cudaFree(d_{b});' for b in input_bufs + output_bufs + inout_bufs]), '    ')}
}}
"""

    with open(host_path, "w") as f:
        f.write(textwrap.dedent(host_code))

    return host_path
