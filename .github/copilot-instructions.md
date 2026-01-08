# Copilot / AI agent instructions for Kernel-AutoValidator

These instructions help an AI coding agent be immediately productive in this repository. They are distilled from the project `README.md` and reflect discoverable structure and patterns.

**Overview:**
- Purpose: automatic, reference-free correctness validation for GPU/CUDA tensor kernels.
- Big idea: infer semantics from memory access patterns and symbolic index expressions (`threadIdx`, `blockIdx`, `blockDim`).

**Where to start (quick):**
- Read the top-level `README.md` to understand goals and limitations. It contains a repository layout and the validation workflow.
- The repository describes these logical components (look for these directories or files):
  - `examples/` — sample CUDA kernels (start here for test inputs)
  - `validator/parser/` — lightweight CUDA parsing (entry point for extraction of memory accesses)
  - `validator/analysis/` — static semantic inference passes (buffer roles, index analysis)
  - `validator/codegen/` — host-side test harness generation
  - `validator/runtime/` — kernel execution and reference-free checks
  - `validator/validate.py` — described as the end-to-end validation entry (try this for manual runs)

**Concrete patterns to look for and use in examples**
- Thread-indexing expressions: patterns like `blockIdx.x * blockDim.x + threadIdx.x` are the primary signal for shape and execution-configuration inference. When you search code, match these arithmetic patterns and symbolic names.
- Buffer role inference: parameters that appear only on the RHS are inputs; parameters written to are outputs; parameters both read and written are in-place. Use read/write sets extracted by the parser as authoritative.
- Safe-shape inference: the validator derives tensor sizes from maxima/minima of index expressions (gridDim * blockDim, plus any explicit bounds). When adding code, keep shape derivation logic centralized in `analysis` to avoid duplication.

**Examples (use these in prompts or tests)**
- Minimal kernel to test pipeline:
  ```cpp
  __global__ void vec_add(float* A, float* B, float* C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    C[i] = A[i] + B[i];
  }
  ```
  The agent should expect: `A,B` = inputs, `C` = output, shape `N = gridDim.x * blockDim.x`.

**Repository-specific development conventions**
- Add new static analysis passes as modules under `validator/analysis/` and keep them pure (take parsed AST or access sets and return analysis objects). Tests or harnesses that exercise a pass belong in `examples/` or `scripts/`.
- Code generation should live in `validator/codegen/` and emit a host-side harness that the `runtime` can compile/execute; prefer small, focused emitted code so validation failures are easy to inspect.
- Runtime checks are property-based (determinism, no-NaNs, data-dependence). When you add a new check, add a short failure message describing the invariant and a reproducible example kernel if possible.

**Integration & testing notes (discoverable guidance)**
- Manual run (try): `python3 validator/validate.py examples/<kernel>.cu` — this reflects the README-described `validate.py` entry point. If `validate.py` accepts different args, inspect its top lines to confirm invocation.
- Use `examples/` kernels as canonical inputs for new features. If adding support for a new pattern, include an `examples/` kernel plus a short script in `scripts/` demonstrating `validate.py` usage.

**What not to change lightly**
- Don’t change the parser heuristics unless absolutely necessary. Many downstream passes assume the parser's memory-access representations.
- Avoid scattering symbolic-index simplification logic across files — centralize it in `validator/analysis/` so other passes reuse consistent forms.

**Pull request guidance for AI agents**
- Provide a short description that references: files changed, new example kernels (path), and a one-line summary of the new invariant or analysis added.
- If a PR adds a new analysis pass, include an `examples/` kernel and a short script or invocation demonstrating that `validate.py` flags or accepts the kernel.

If anything here is unclear or if you want the agent to include runnable CLI examples that probe `validate.py`, tell me and I will inspect `validator/validate.py` (and other files) and iterate the instructions.
