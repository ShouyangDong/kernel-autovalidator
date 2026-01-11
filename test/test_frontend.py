from ksv.frontend.preprocess import preprocess_cuda
from ksv.frontend.ast_builder import ASTBuilder

code = r"""
__global__ void kernel(float* A) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  A[i] = 0;
}
"""

pre = preprocess_cuda(code)
ast = ASTBuilder().build(pre)

ast.show()
