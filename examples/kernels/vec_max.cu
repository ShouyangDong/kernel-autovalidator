__global__ void max_dev(const float *__restrict__ input,
                        float *__restrict__ output) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= 128)
    return;

  float max_val = -FLT_MAX;
  for (int col = 0; col < 256; col++) {
    int idx = row * 256 + col;
    max_val = fmaxf(max_val, input[idx]);
  }
  output[row] = max_val;
}
