from dataclasses import dataclass

from pycparser import c_ast


@dataclass
class MemoryAccess:
    buffer: str  # buffer name
    access_type: str  # "load" or "store"
    index_expr: c_ast.Node  # index AST
    node: c_ast.Node  # original AST node (for debugging)
