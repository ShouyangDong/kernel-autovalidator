import os
import subprocess


def run_kernel(host_path):
    assert host_path.endswith(".cu")
    workdir = os.path.dirname(os.path.abspath(host_path))
    binary_path = os.path.join(workdir, "a.out")

    # Compile
    compile_cmd = [
        "nvcc",
        host_path,
        "-O2",
        "-std=c++14",
        "-o",
        binary_path,
    ]

    try:
        subprocess.check_output(compile_cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("CUDA compilation failed:\n" + e.output.decode())

    # Run
    try:
        output = subprocess.check_output(
            [binary_path],
            stderr=subprocess.STDOUT,
            cwd=workdir,
        ).decode()
    except subprocess.CalledProcessError as e:
        raise RuntimeError("CUDA execution failed:\n" + e.output.decode())

    return parse_outputs(output)
