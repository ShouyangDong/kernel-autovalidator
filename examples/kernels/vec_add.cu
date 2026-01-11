__global__ void vec_add(float* A, float* B, float* C) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 1024) {
    C[i] = A[i] + B[i];
  }
}
