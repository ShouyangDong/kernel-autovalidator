from pycparser import c_ast


class SymbolCollector(c_ast.NodeVisitor):
    def __init__(self):
        self.symbols = set()

    def visit_ID(self, node):
        self.symbols.add(node.name)

    def visit_StructRef(self, node):
        # threadIdx.x / blockIdx.x
        if isinstance(node.name, c_ast.ID):
            full = f"{node.name.name}.{node.field.name}"
            self.symbols.add(full)


def extract_index_expressions(accesses):
    """
    Extract symbolic index expressions from memory accesses.
    """
    index_exprs = []

    for acc in accesses:
        # `acc` is a MemoryAccess object (ksv.ir.memory_access.MemoryAccess)
        # with fields: buffer, access_type, index_expr, node.
        index_ast = getattr(acc, "index_expr", None)
        if index_ast is None:
            continue

        collector = SymbolCollector()
        # index_expr may be a single AST node or a list (for multi-dimensional
        # access)
        if isinstance(index_ast, list):
            for sub in index_ast:
                collector.visit(sub)
        else:
            collector.visit(index_ast)

        index_exprs.append(
            {
                "buffer": getattr(acc, "buffer", None),
                "access_type": getattr(acc, "access_type", None),
                "expr_ast": index_ast,
                "symbols": collector.symbols,
            }
        )

    return index_exprs
