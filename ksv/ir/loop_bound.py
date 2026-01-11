from dataclasses import dataclass

from pycparser import c_ast


@dataclass
class LoopBound:
    var: str
    lower: c_ast.Node  # usually Constant or ID
    upper: c_ast.Node  # exclusive upper bound
