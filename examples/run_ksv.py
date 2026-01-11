# scripts/run_ksv.py
import argparse
from ksv.pipeline import run_ksv


def main():
    parser = argparse.ArgumentParser(
        description="Run Kernel Semantic Validation (KSV) on a CUDA kernel"
    )
    parser.add_argument(
        "kernel_path",
        type=str,
        help="Path to the CUDA kernel (.cu) file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    run_ksv(
        kernel_path=args.kernel_path,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
