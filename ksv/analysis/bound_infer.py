# ksv/analysis/infer_tensor_shapes.py

from typing import Dict, List

from pycparser import c_ast

from ksv.ir.memory_access import MemoryAccess


def eval_upper_bound(expr, var_bounds):
    """
    Compute conservative upper bound of an index expression.
    """

    # Helper: evaluate an AST node to an integer when possible. For ID nodes,
    # attempt to resolve using var_bounds which may contain LoopBound objects
    # or tuples (lower, upper).
    def node_to_int(node):
        if isinstance(node, c_ast.Constant):
            return int(node.value)

        if isinstance(node, c_ast.ID):
            name = node.name
            if name not in var_bounds:
                raise ValueError(f"Unknown variable {name}")
            vb = var_bounds[name]
            # vb may be a LoopBound dataclass or a (lower, upper) tuple
            if hasattr(vb, "upper"):
                return node_to_int(vb.upper)
            # assume tuple (lower, upper) where upper is int
            _, ub = vb
            return int(ub)

        if isinstance(node, c_ast.BinaryOp):
            lhs = node_to_int(node.left)
            rhs = node_to_int(node.right)
            if node.op == "+":
                return lhs + rhs
            if node.op == "-":
                return lhs - rhs
            if node.op == "*":
                return lhs * rhs

        raise NotImplementedError(f"Unsupported expr {type(node)}")

    # Main eval_upper_bound behavior: compute maximum index (inclusive)
    if isinstance(expr, c_ast.Constant):
        return int(expr.value)

    if isinstance(expr, c_ast.ID):
        var = expr.name
        if var in var_bounds:
            vb = var_bounds[var]
            if hasattr(vb, "upper"):
                ub_node = vb.upper
                ub_val = node_to_int(ub_node)
            else:
                _, ub_val = vb
            # upper is exclusive, so max index = ub_val - 1
            return ub_val - 1
        raise ValueError(f"Unknown variable {var}")

    if isinstance(expr, c_ast.BinaryOp):
        lhs = eval_upper_bound(expr.left, var_bounds)
        rhs = eval_upper_bound(expr.right, var_bounds)

        if expr.op == "+":
            return lhs + rhs
        if expr.op == "-":
            return lhs - rhs
        if expr.op == "*":
            return lhs * rhs

    raise NotImplementedError(f"Unsupported expr {type(expr)}")


def infer_tensor_shapes(
    memory_accesses: List[MemoryAccess],
    loop_bounds: Dict[str, tuple],
    thread_bounds: Dict[str, tuple],
):
    """
    Infer minimal safe buffer sizes for all tensors.
    """
    # 合并变量上界
    var_bounds = {}
    var_bounds.update(loop_bounds)
    var_bounds.update(thread_bounds)

    buffer_max_index = {}

    for acc in memory_accesses:
        ub = eval_upper_bound(acc.index_expr, var_bounds)

        if acc.buffer not in buffer_max_index:
            buffer_max_index[acc.buffer] = ub
        else:
            buffer_max_index[acc.buffer] = max(
                buffer_max_index[acc.buffer], ub
            )

    # size = max_index + 1
    buffer_shapes = {
        buf: max_idx + 1 for buf, max_idx in buffer_max_index.items()
    }

    return buffer_shapes
