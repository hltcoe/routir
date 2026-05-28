#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${HOME}/.conda/envs/search-service/bin/python3"

"${PYTHON}" -m grpc_tools.protoc \
    -Isrc/routir/proto \
    --python_out=src/routir/proto/_generated \
    --grpc_python_out=src/routir/proto/_generated \
    --pyi_out=src/routir/proto/_generated \
    routir.proto

# Rewrite the bare `import routir_pb2` in routir_pb2_grpc.py to a relative
# import so the generated package can live under routir.proto._generated.
sed -i 's/^import routir_pb2 as routir__pb2$/from . import routir_pb2 as routir__pb2/' \
    src/routir/proto/_generated/routir_pb2_grpc.py

echo "Generated routir_pb2.py, routir_pb2_grpc.py, routir_pb2.pyi in src/routir/proto/_generated/"
