# ksv/analysis/loop_bounds.py

from typing import Dict

from pycparser import c_ast

from ksv.ir.loop_bound import LoopBound


class LoopBoundExtractor(c_ast.NodeVisitor):
    """
    Extract loop variable bounds from for-loops.
    """

    def __init__(self):
        self.bounds: Dict[str, LoopBound] = {}

    # ------------------------------------------------------------
    # for-loop handling
    # ------------------------------------------------------------
    def visit_For(self, node: c_ast.For):
        """
        Handle patterns like:
        for (int i = 0; i < N; i++)
        for (i = 0; i < N; i++)
        """
        var, lower = self._parse_init(node.init)
        upper = self._parse_cond(node.cond, var)

        if var is not None and lower is not None and upper is not None:
            self.bounds[var] = LoopBound(var=var, lower=lower, upper=upper)

        self.generic_visit(node)

    def visit_Assignment(self, node: c_ast.Assignment):
        # Detect assignments of the form: row = blockIdx.x * blockDim.x + threadIdx.x
        # and infer an upper bound: gridDim.x * blockDim.x
        l = node.lvalue
        r = node.rvalue
        if isinstance(l, c_ast.ID) and self._expr_has_thread_index(r):
            # build gridDim.x * blockDim.x as exclusive upper bound
            grid_x = c_ast.StructRef(c_ast.ID("gridDim"), ".", c_ast.ID("x"))
            block_x = c_ast.StructRef(c_ast.ID("blockDim"), ".", c_ast.ID("x"))
            upper = c_ast.BinaryOp("*", grid_x, block_x)
            if l.name not in self.bounds:
                self.bounds[l.name] = LoopBound(
                    var=l.name, lower=c_ast.Constant("int", "0"), upper=upper
                )

        self.generic_visit(node)

    # ------------------------------------------------------------
    # If-statement guards: e.g., if (row >= N) return;
    # ------------------------------------------------------------
    def visit_If(self, node: c_ast.If):
        # Case A: guard that returns when out-of-range, e.g., `if (row >= N)
        # return;`
        if self._contains_return(node.iftrue):
            # parse simple binary comparisons like `row >= N` or `row > N`
            cond = node.cond
            if isinstance(cond, c_ast.BinaryOp):
                # check patterns where one side is an ID
                left = cond.left
                right = cond.right
                # case: ID op CONST/ID
                if isinstance(left, c_ast.ID):
                    var = left.name
                    ub = self._bound_from_guard_op(cond.op, right)
                    if ub is not None:
                        # conservative: set upper if not present
                        if var not in self.bounds:
                            self.bounds[var] = LoopBound(
                                var=var,
                                lower=c_ast.Constant("int", "0"),
                                upper=ub,
                            )
                # case: CONST/ID op ID  (e.g., N <= row) -> flip
                elif isinstance(right, c_ast.ID):
                    var = right.name
                    ub = self._bound_from_guard_op(
                        self._flip_op(cond.op), left
                    )
                    if ub is not None:
                        if var not in self.bounds:
                            self.bounds[var] = LoopBound(
                                var=var,
                                lower=c_ast.Constant("int", "0"),
                                upper=ub,
                            )

        # Case B: simple if-guard that bounds execution, e.g., `if (i < N) { ... }`
        # Interpret this as `i` has exclusive upper bound `N` within the
        # guarded block.
        cond = node.cond
        if isinstance(cond, c_ast.BinaryOp):
            left = cond.left
            right = cond.right
            if isinstance(left, c_ast.ID) and isinstance(
                right, (c_ast.Constant, c_ast.ID)
            ):
                if cond.op == "<":
                    ub = right
                elif cond.op == "<=":
                    # exclusive upper bound = right + 1
                    ub = c_ast.BinaryOp("+", right, c_ast.Constant("int", "1"))
                else:
                    ub = None

                if ub is not None:
                    var = left.name
                    # If there is an existing bound, prefer a constant tighter
                    # bound
                    if var not in self.bounds:
                        self.bounds[var] = LoopBound(
                            var=var, lower=c_ast.Constant("int", "0"), upper=ub
                        )
                    else:
                        existing = self.bounds[var].upper
                        # If new upper is a Constant and existing is not,
                        # replace (tighter)
                        if isinstance(ub, c_ast.Constant) and not isinstance(
                            existing, c_ast.Constant
                        ):
                            self.bounds[var] = LoopBound(
                                var=var,
                                lower=c_ast.Constant("int", "0"),
                                upper=ub,
                            )
                        # If both are constants, pick the smaller numeric upper
                        elif isinstance(ub, c_ast.Constant) and isinstance(
                            existing, c_ast.Constant
                        ):
                            try:
                                new_val = int(ub.value)
                                existing_val = int(existing.value)
                                if new_val < existing_val:
                                    self.bounds[var] = LoopBound(
                                        var=var,
                                        lower=c_ast.Constant("int", "0"),
                                        upper=ub,
                                    )
                            except Exception:
                                pass

        self.generic_visit(node)

    def _contains_return(self, node):
        if node is None:
            return False
        # direct Return
        if isinstance(node, c_ast.Return):
            return True
        # block of statements
        for _, child in node.children():
            if self._contains_return(child):
                return True
        return False

    def _flip_op(self, op: str) -> str:
        # flip sides: a < b  ->  b > a  same semantics for our purposes
        flips = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}
        return flips.get(op, op)

    def _bound_from_guard_op(self, op: str, rhs):
        """Given a guard `var op rhs` that leads to `return` when true,
        compute the exclusive upper bound for var (as a c_ast.Node).
        Examples:
          var >= C  -> upper = C
          var > C   -> upper = C + 1
        Returns None if unsupported.
        """
        # only support when rhs is a Constant or ID
        if not isinstance(rhs, (c_ast.Constant, c_ast.ID)):
            return None

        if op == ">=":
            return rhs
        if op == ">":
            # produce rhs + 1
            return c_ast.BinaryOp("+", rhs, c_ast.Constant("int", "1"))
        # if guard is var < C and return when true, that would imply lower
        # bound, skip
        return None

    # ------------------------------------------------------------
    # Parse init: int i = 0
    # ------------------------------------------------------------
    def _parse_init(self, init):
        if isinstance(init, c_ast.Decl):
            if init.init is None:
                return None, None
            return init.name, init.init

        if isinstance(init, c_ast.Assignment):
            if isinstance(init.lvalue, c_ast.ID):
                return init.lvalue.name, init.rvalue

        return None, None

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def _expr_has_thread_index(self, expr) -> bool:
        """Return True if expression contains thread/block/grid references.

        Looks for `threadIdx`, `blockIdx`, `blockDim`, or `gridDim` used as
        struct references (e.g., `threadIdx.x`) anywhere inside `expr`.
        """
        if expr is None:
            return False

        stack = [expr]
        while stack:
            n = stack.pop()
            if isinstance(n, c_ast.StructRef):
                # n.name is typically an ID (e.g., threadIdx)
                if isinstance(n.name, c_ast.ID) and n.name.name in (
                    "threadIdx",
                    "blockIdx",
                    "blockDim",
                    "gridDim",
                ):
                    return True
                # also continue scanning children
            if isinstance(n, c_ast.ID):
                # rare: direct ID usage
                if n.name in ("threadIdx", "blockIdx", "blockDim", "gridDim"):
                    return True

            for _, child in n.children():
                stack.append(child)

        return False

    # ------------------------------------------------------------
    # Parse condition: i < N, i <= N
    # ------------------------------------------------------------
    def _parse_cond(self, cond, var):
        if not isinstance(cond, c_ast.BinaryOp):
            return None

        if cond.left.__class__ != c_ast.ID:
            return None
        if cond.left.name != var:
            return None

        # i < N
        if cond.op == "<":
            return cond.right

        # i <= N  → conservative: N+1
        if cond.op == "<=":
            return c_ast.BinaryOp("+", cond.right, c_ast.Constant("int", "1"))

        return None

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def extract(self):
        return self.bounds


def extract_loop_bounds(kernel_func: c_ast.FuncDef):
    extractor = LoopBoundExtractor()
    extractor.visit(kernel_func)
    return extractor.extract()
