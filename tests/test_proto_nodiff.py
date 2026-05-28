"""Guard against stale gRPC stubs.

Regenerates ``routir_pb2.py``, ``routir_pb2_grpc.py``, ``routir_pb2.pyi``
from ``src/routir/proto/routir.proto`` into a temp directory and compares
byte-for-byte against the checked-in stubs.  Fails loudly if anyone edits
the proto without rerunning ``scripts/build_proto.sh``.
"""

import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
PROTO_DIR = REPO_ROOT / "src" / "routir" / "proto"
PROTO_FILE = PROTO_DIR / "routir.proto"
GENERATED_DIR = PROTO_DIR / "_generated"
EXPECTED_FILES = ("routir_pb2.py", "routir_pb2_grpc.py", "routir_pb2.pyi")


def _python_with_grpc_tools() -> str:
    """Pick an interpreter that has grpc_tools.protoc available.

    Prefers ``sys.executable`` so the test runs in whatever env the user
    invokes pytest under, but falls back to the project's pinned conda env
    when the runner doesn't have ``grpcio-tools`` installed.
    """
    candidates = [
        sys.executable,
        str(Path.home() / ".conda" / "envs" / "search-service" / "bin" / "python3"),
    ]
    for python in candidates:
        try:
            subprocess.run(
                [python, "-c", "import grpc_tools.protoc"],
                check=True,
                capture_output=True,
            )
            return python
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    return ""


def test_generated_stubs_match_proto(tmp_path):
    python = _python_with_grpc_tools()
    if not python:
        pytest.skip("grpc_tools not installed")

    out = tmp_path
    result = subprocess.run(
        [
            python, "-m", "grpc_tools.protoc",
            f"-I={PROTO_DIR}",
            f"--python_out={out}",
            f"--grpc_python_out={out}",
            f"--pyi_out={out}",
            "routir.proto",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(
            f"grpc_tools.protoc failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Apply the same rewrite ``scripts/build_proto.sh`` does so the relative
    # import matches the checked-in file.
    grpc_stub = out / "routir_pb2_grpc.py"
    grpc_stub.write_text(
        grpc_stub.read_text().replace(
            "import routir_pb2 as routir__pb2",
            "from . import routir_pb2 as routir__pb2",
        )
    )

    for name in EXPECTED_FILES:
        generated = (out / name).read_bytes()
        checked_in = (GENERATED_DIR / name).read_bytes()
        if generated != checked_in:
            pytest.fail(
                f"{name} is stale; run scripts/build_proto.sh and commit"
            )
