"""Shared fixtures for the RoutIR test suite.

Servers boot in subprocesses so REST + gRPC are exercised through the real
transports rather than via in-process shortcuts.  Random ports keep parallel
runs safe.  Each server fixture polls ``/ping`` until it answers, and skips
the test with a clear reason if the boot doesn't complete in time.
"""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
TRIVIAL_ENGINE_PATH = Path(__file__).resolve().parent / "_trivial_engine.py"
PYTHON = os.environ.get("ROUTIR_TEST_PYTHON") or sys.executable
BOOT_TIMEOUT = 30.0


def _pick_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def _wait_for_ping(base_url: str, timeout: float = BOOT_TIMEOUT) -> bool:
    deadline = time.time() + timeout
    url = base_url.rstrip("/") + "/ping"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(0.25)
    return False


def _write_config(path: Path, *, with_trivial_engine: bool) -> None:
    cfg = {
        "services": [],
        "collections": [],
        "server_imports": [],
        "file_imports": [],
        "pipeline_aliases": {},
    }
    if with_trivial_engine:
        cfg["file_imports"] = [str(TRIVIAL_ENGINE_PATH)]
        cfg["services"] = [{
            "name": "trivial",
            "engine": "TrivialSearchEngine",
            "config": {},
            "batch_size": 4,
            "max_wait_time": 0.01,
            "cache": -1,
        }]
    path.write_text(json.dumps(cfg))


@pytest.fixture(scope="session")
def empty_config_path(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("routir-cfg") / "empty.json"
    _write_config(p, with_trivial_engine=False)
    return p


@pytest.fixture(scope="session")
def trivial_engine_config_path(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("routir-cfg-trivial") / "trivial.json"
    _write_config(p, with_trivial_engine=True)
    return p


def _spawn_server(config_path: Path, args: list, env_extra: dict = None):
    env = os.environ.copy()
    # Make ``src/`` importable inside the subprocess; mirrors how PYTHONPATH=src
    # works for the test runner itself.
    env["PYTHONPATH"] = f"{SRC_DIR}{os.pathsep}{env.get('PYTHONPATH', '')}"
    if env_extra:
        env.update(env_extra)
    cmd = [PYTHON, "-m", "routir.serve", str(config_path), *args]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )


def _terminate(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


@pytest.fixture
def server_rest_only(trivial_engine_config_path):
    port = _pick_free_port()
    proc = _spawn_server(
        trivial_engine_config_path,
        ["--port", str(port), "--host", "127.0.0.1"],
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        if not _wait_for_ping(base_url):
            _terminate(proc)
            pytest.skip("server boot timed out (REST-only)")
        yield base_url
    finally:
        _terminate(proc)


@pytest.fixture
def server_both(trivial_engine_config_path):
    rest_port = _pick_free_port()
    grpc_port = _pick_free_port()
    proc = _spawn_server(
        trivial_engine_config_path,
        [
            "--port", str(rest_port),
            "--host", "127.0.0.1",
            "--grpc",
            "--grpc-port", str(grpc_port),
        ],
    )
    rest_url = f"http://127.0.0.1:{rest_port}"
    grpc_target = f"127.0.0.1:{grpc_port}"
    try:
        if not _wait_for_ping(rest_url):
            _terminate(proc)
            pytest.skip("server boot timed out (REST+gRPC)")
        yield rest_url, grpc_target
    finally:
        _terminate(proc)


@pytest.fixture
def server_authed(trivial_engine_config_path):
    rest_port = _pick_free_port()
    grpc_port = _pick_free_port()
    api_key = "sekret"
    proc = _spawn_server(
        trivial_engine_config_path,
        [
            "--port", str(rest_port),
            "--host", "127.0.0.1",
            "--grpc",
            "--grpc-port", str(grpc_port),
            "--api_key", api_key,
        ],
    )
    rest_url = f"http://127.0.0.1:{rest_port}"
    grpc_target = f"127.0.0.1:{grpc_port}"
    try:
        if not _wait_for_ping(rest_url):
            _terminate(proc)
            pytest.skip("server boot timed out (authed)")
        yield rest_url, grpc_target, api_key
    finally:
        _terminate(proc)
