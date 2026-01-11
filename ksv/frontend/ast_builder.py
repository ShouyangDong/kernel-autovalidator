# ksv/frontend/ast_builder.py

from pycparser import c_ast, c_parser
from pycparser.plyparser import ParseError


class ASTBuilder:
    """
    Build a pycparser AST from preprocessed C-like code.
    """

    def __init__(self):
        self.parser = c_parser.CParser()

    def build(self, code: str) -> c_ast.FileAST:
        """
        Parse code into a pycparser AST.

        Args:
            code (str): Preprocessed C-like source code

        Returns:
            c_ast.FileAST: Root of the AST
        """
        try:
            ast = self.parser.parse(code)
        except ParseError as e:
            raise RuntimeError(f"AST parsing failed: {e}")

        return ast.ext[0]


def build_ast(src_code):
    ast = ASTBuilder().build(src_code)
    return ast
