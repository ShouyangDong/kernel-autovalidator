# ksv/analysis/buffer_roles.py

from collections import defaultdict

from pycparser import c_ast


class BufferRoleInferencer(c_ast.NodeVisitor):
    """
    Infer input / output roles for pointer-typed kernel arguments.
    """

    def __init__(self, kernel_func: c_ast.FuncDef):
        self.kernel_func = kernel_func
        # buffer_name -> {"load": bool, "store": bool}
        self.buffer_access = defaultdict(
            lambda: {"load": False, "store": False}
        )

        # pointer-typed kernel args
        self.pointer_args = set()

        self._collect_pointer_args()

    # ------------------------------------------------------------
    # Step 1: collect pointer kernel arguments
    # ------------------------------------------------------------
    def _collect_pointer_args(self):
        func_decl = self.kernel_func.decl.type
        if not isinstance(func_decl, c_ast.FuncDecl):
            return

        if func_decl.args is None:
            return

        for param in func_decl.args.params:
            if self._is_pointer_type(param.type):
                self.pointer_args.add(param.name)

    def _is_pointer_type(self, ty):
        return isinstance(ty, c_ast.PtrDecl)

    # ------------------------------------------------------------
    # Step 2: visit memory accesses
    # ------------------------------------------------------------
    def visit_Assignment(self, node: c_ast.Assignment):
        """
        Detect stores: buffer[idx] = ...
        """
        # LHS store
        if isinstance(node.lvalue, c_ast.ArrayRef):
            base = self._get_array_base_name(node.lvalue)
            if base in self.pointer_args:
                self.buffer_access[base]["store"] = True

        # RHS load
        self._visit_rhs(node.rvalue)
        self.generic_visit(node)

    def visit_Decl(self, node: c_ast.Decl):
        """
        Detect loads in initialization:
            float x = buffer[idx];
        """
        if node.init is not None:
            self._visit_rhs(node.init)
        self.generic_visit(node)

    def _visit_rhs(self, node):
        """
        Recursively detect buffer loads in RHS expressions.
        """
        for child in node.children():
            _, child_node = child
            if isinstance(child_node, c_ast.ArrayRef):
                base = self._get_array_base_name(child_node)
                if base in self.pointer_args:
                    self.buffer_access[base]["load"] = True
            self._visit_rhs(child_node)

    # ------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------
    def _get_array_base_name(self, array_ref: c_ast.ArrayRef):
        """
        Extract base identifier name from A[i][j] or A[i]
        """
        node = array_ref.name
        while isinstance(node, c_ast.ArrayRef):
            node = node.name
        if isinstance(node, c_ast.ID):
            return node.name
        return None

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def infer(self):
        """
        Returns:
            dict: buffer_name -> role ("input", "output", "inout", "unused")
        """
        self.visit(self.kernel_func)

        roles = {}
        for buf in self.pointer_args:
            acc = self.buffer_access[buf]
            if acc["load"] and acc["store"]:
                roles[buf] = "inout"
            elif acc["load"]:
                roles[buf] = "input"
            elif acc["store"]:
                roles[buf] = "output"
            else:
                roles[buf] = "unused"

        return roles


def infer_buffer_roles(kernel_func: c_ast.FuncDef):
    """
    Convenience wrapper.
    """
    return BufferRoleInferencer(kernel_func).infer()
