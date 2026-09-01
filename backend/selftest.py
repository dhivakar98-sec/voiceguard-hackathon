"""
End-to-end self test — run this after unzipping to prove the app works here.

    python backend/selftest.py            # uses the venv python that runs it

It starts the real server on a free port, waits for /api/health, uploads every
clip in sample_audio/ through /api/analyze, prints the verdicts, and shuts down.
Uses only the standard library plus the core deps, so it works in the minimal
install with no ML packages present.

Exit code 0 = the package is good to ship.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
ROOT = BACKEND_DIR.parent
SAMPLES = ROOT / "sample_audio"

GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"
if os.name == "nt" and not os.environ.get("WT_SESSION"):
    GREEN = RED = DIM = RESET = ""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def get_json(url: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def post_file(url: str, path: Path, timeout: float = 180.0) -> tuple[int, dict]:
    """Minimal multipart/form-data POST — no requests/httpx needed."""
    boundary = f"----VoiceGuard{uuid.uuid4().hex}"
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'.encode(),
            b"Content-Type: application/octet-stream\r\n\r\n",
            path.read_bytes(),
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode() or "{}")


def main() -> int:
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    print(f"{DIM}Starting the server on {base} ...{RESET}")

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--app-dir", str(BACKEND_DIR),
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    failures: list[str] = []
    try:
        # ---- 1. server comes up ------------------------------------------
        health = None
        for _ in range(120):
            if proc.poll() is not None:
                print(f"{RED}The server exited immediately:{RESET}\n{proc.stdout.read()}")
                return 1
            try:
                health = get_json(f"{base}/api/health", timeout=3)
                break
            except Exception:
                time.sleep(0.5)
        if health is None:
            failures.append("the server never answered /api/health")
            print(f"{RED}FAIL{RESET}  /api/health never answered")
            return 1

        # give a downloading ML model a chance, but never block forever
        waited = 0.0
        while health.get("status") == "loading" and waited < 600:
            time.sleep(3); waited += 3
            print(f"{DIM}  detector still loading ({waited:.0f}s) ...{RESET}")
            health = get_json(f"{base}/api/health", timeout=5)

        print(f"{GREEN}PASS{RESET}  /api/health  backend={health['backend']} model={health['model']}")
        if health["backend"] not in {"ml", "heuristic"}:
            failures.append(f"unexpected backend {health['backend']!r}")
        if health["backend"] == "heuristic" and health.get("ml_load_error"):
            print(f"{DIM}      (ML unavailable: {str(health['ml_load_error'])[:110]}){RESET}")

        # ---- 2. frontend is served ---------------------------------------
        with urllib.request.urlopen(f"{base}/", timeout=10) as resp:
            html = resp.read().decode("utf-8", "replace")
        if resp.status == 200 and "VoiceGuard" in html:
            print(f"{GREEN}PASS{RESET}  GET /            frontend served ({len(html)} bytes)")
        else:
            failures.append("the frontend was not served at /")
            print(f"{RED}FAIL{RESET}  GET /            unexpected response")

        # ---- 3. every sample clip analyses -------------------------------
        clips = sorted(
            p for p in SAMPLES.glob("*")
            if p.suffix.lower() in {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aiff"}
        ) if SAMPLES.is_dir() else []
        if not clips:
            print(f"{RED}FAIL{RESET}  no clips found in sample_audio/")
            failures.append("sample_audio/ is empty")
        for clip in clips:
            status, data = post_file(f"{base}/api/analyze", clip)
            if status == 200 and data.get("verdict") in {"HUMAN", "FAKE"}:
                print(
                    f"{GREEN}PASS{RESET}  POST /api/analyze  {clip.name:28s} "
                    f"-> {data['verdict']:5s} {data['confidence']:3d}/100 "
                    f"(p={data['spoof_probability']}, {data['backend']}, {data['processing_ms']} ms)"
                )
            else:
                print(f"{RED}FAIL{RESET}  POST /api/analyze  {clip.name}: "
                      f"HTTP {status} {data.get('detail', data)}")
                failures.append(f"{clip.name} did not analyse")

        # ---- 4. bad input is rejected, not crashed on --------------------
        junk = Path(ROOT / "_selftest_junk.wav")
        junk.write_bytes(b"this is definitely not audio" * 40)
        try:
            status, data = post_file(f"{base}/api/analyze", junk)
            if status == 400 and data.get("detail"):
                print(f"{GREEN}PASS{RESET}  corrupt upload rejected with a clear 400")
            else:
                print(f"{RED}FAIL{RESET}  corrupt upload returned HTTP {status}: {data}")
                failures.append("corrupt upload was not handled cleanly")
        finally:
            junk.unlink(missing_ok=True)

        # ---- 5. server survived everything ------------------------------
        if proc.poll() is None and get_json(f"{base}/api/health", timeout=5):
            print(f"{GREEN}PASS{RESET}  server still healthy after the bad upload")
        else:
            failures.append("the server died during the test")
            print(f"{RED}FAIL{RESET}  the server is no longer responding")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    print()
    if failures:
        print(f"{RED}{len(failures)} check(s) failed:{RESET}")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"{GREEN}All checks passed — this package runs on this machine.{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
