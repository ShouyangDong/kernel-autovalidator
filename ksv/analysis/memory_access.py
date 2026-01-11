# ksv/analysis/memory_access.py

from typing import List

from pycparser import c_ast

from ksv.ir.memory_access import MemoryAccess


class MemoryAccessExtractor(c_ast.NodeVisitor):
    """
    Extract global memory load/store accesses from a kernel AST.
    """

    def __init__(self, kernel_func: c_ast.FuncDef, pointer_args):
        self.kernel_func = kernel_func
        self.pointer_args = set(pointer_args)
        self.accesses: List[MemoryAccess] = []

    # ------------------------------------------------------------
    # Assignment: store on LHS, load on RHS
    # ------------------------------------------------------------
    def visit_Assignment(self, node: c_ast.Assignment):
        # Store: buffer[idx] = ...
        if isinstance(node.lvalue, c_ast.ArrayRef):
            buf, idx = self._parse_array_ref(node.lvalue)
            if buf in self.pointer_args:
                self.accesses.append(
                    MemoryAccess(
                        buffer=buf,
                        access_type="store",
                        index_expr=idx,
                        node=node.lvalue,
                    )
                )

        # Loads on RHS
        self._visit_rhs(node.rvalue)
        self.generic_visit(node)

    # ------------------------------------------------------------
    # Initialization: float x = buffer[idx]
    # ------------------------------------------------------------
    def visit_Decl(self, node: c_ast.Decl):
        if node.init is not None:
            self._visit_rhs(node.init)
        self.generic_visit(node)

    # ------------------------------------------------------------
    # RHS traversal
    # ------------------------------------------------------------
    def _visit_rhs(self, node):
        if node is None:
            return

        if isinstance(node, c_ast.ArrayRef):
            buf, idx = self._parse_array_ref(node)
            if buf in self.pointer_args:
                self.accesses.append(
                    MemoryAccess(
                        buffer=buf,
                        access_type="load",
                        index_expr=idx,
                        node=node,
                    )
                )

        for _, child in node.children():
            self._visit_rhs(child)

    # ------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------
    def _parse_array_ref(self, node: c_ast.ArrayRef):
        """
        For A[i][j] flatten as base=A, index=(i,j) nested AST.
        """
        indices = []
        cur = node
        while isinstance(cur, c_ast.ArrayRef):
            indices.append(cur.subscript)
            cur = cur.name

        if not isinstance(cur, c_ast.ID):
            return None, None

        indices.reverse()
        index_expr = indices[0] if len(indices) == 1 else indices
        return cur.name, index_expr

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def extract(self) -> List[MemoryAccess]:
        self.visit(self.kernel_func)
        return self.accesses


def extract_memory_accesses(kernel_func: c_ast.FuncDef, pointer_args):
    """
    Returns a list of MemoryAccess records.
    """
    extractor = MemoryAccessExtractor(kernel_func, pointer_args)
    return extractor.extract()
