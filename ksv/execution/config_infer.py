import math

from pycparser import c_ast


class ThreadUsageVisitor(c_ast.NodeVisitor):
    def __init__(self):
        self.uses_thread_x = False
        self.uses_thread_y = False
        self.uses_thread_z = False
        self.uses_block_x = False
        self.uses_block_y = False
        self.uses_block_z = False

    def visit_StructRef(self, node):
        if isinstance(node.name, c_ast.ID) and isinstance(node.field, c_ast.ID):
            if node.name.name == "threadIdx":
                if node.field.name == "x":
                    self.uses_thread_x = True
                elif node.field.name == "y":
                    self.uses_thread_y = True
                elif node.field.name == "z":
                    self.uses_thread_z = True
            if node.name.name == "blockIdx":
                if node.field.name == "x":
                    self.uses_block_x = True
                elif node.field.name == "y":
                    self.uses_block_y = True
                elif node.field.name == "z":
                    self.uses_block_z = True

    def visit_ID(self, node):
        # support preprocessed names like threadIdx_x / blockIdx_x
        if node.name in ("threadIdx_x",):
            self.uses_thread_x = True
        if node.name in ("threadIdx_y",):
            self.uses_thread_y = True
        if node.name in ("threadIdx_z",):
            self.uses_thread_z = True
        if node.name in ("blockIdx_x",):
            self.uses_block_x = True
        if node.name in ("blockIdx_y",):
            self.uses_block_y = True
        if node.name in ("blockIdx_z",):
            self.uses_block_z = True


def infer_execution_config(kernel_ast, thread_bounds):
    """
    Infer grid/block configuration for kernel launch.
    """
    visitor = ThreadUsageVisitor()
    visitor.visit(kernel_ast)

    # Default 1D kernel
    block_x = block_y = block_z = 1
    grid_x = grid_y = grid_z = 1

    if (visitor.uses_thread_x or visitor.uses_block_x or
        visitor.uses_thread_y or visitor.uses_block_y or
        visitor.uses_thread_z or visitor.uses_block_z):
        # total logical threads needed
        total_threads = 1

        def eval_node_to_int(node, var_bounds):
            if isinstance(node, c_ast.Constant):
                return int(node.value)
            if isinstance(node, c_ast.BinaryOp):
                lhs = eval_node_to_int(node.left, var_bounds)
                rhs = eval_node_to_int(node.right, var_bounds)
                if node.op == '+':
                    return lhs + rhs
                if node.op == '-':
                    return lhs - rhs
                if node.op == '*':
                    return lhs * rhs
            if isinstance(node, c_ast.ID):
                name = node.name
                if name in var_bounds:
                    vb = var_bounds[name]
                    if hasattr(vb, 'upper'):
                        return eval_node_to_int(vb.upper, var_bounds)
                    else:
                        # assume tuple (low, up)
                        return int(vb[1])
            # unknown structure (e.g., gridDim.x * blockDim.x), return conservative default
            return 1024

        for _, vb in thread_bounds.items():
            # vb may be LoopBound or tuple (low, up)
            if hasattr(vb, 'upper'):
                ub_node = vb.upper
                try:
                    ub_val = eval_node_to_int(ub_node, thread_bounds)
                except Exception:
                    ub_val = 1024
            else:
                _, ub_val = vb

            total_threads *= max(1, int(ub_val))

        # Heuristic: choose block/grid per-dimension based on which thread dims are used
        # If only x used: use 1D launch
        use_x = visitor.uses_thread_x or visitor.uses_block_x
        use_y = visitor.uses_thread_y or visitor.uses_block_y
        use_z = visitor.uses_thread_z or visitor.uses_block_z

        if use_x and not use_y and not use_z:
            block_x = min(256, total_threads)
            grid_x = math.ceil(total_threads / block_x)
        elif use_x and use_y and not use_z:
            # spread work across x and y: pick modest y dimension
            block_x = min(64, total_threads)
            block_y = min(16, max(1, math.ceil(total_threads / block_x)))
            grid_x = math.ceil(total_threads / (block_x * block_y))
        else:
            # use x,y,z (or x+z) heuristics
            block_x = min(32, total_threads)
            block_y = min(8, max(1, math.ceil(total_threads / block_x)))
            block_z = min(4, max(1, math.ceil(total_threads / (block_x * block_y))))
            grid_x = math.ceil(total_threads / (block_x * block_y * block_z))

        # (no additional override; keep block_x/block_y/block_z as computed)

    return {
        "block": (block_x, block_y, block_z),
        "grid": (grid_x, grid_y, grid_z),
    }
