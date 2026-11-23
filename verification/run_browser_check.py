"""Bounded local browser QA using installed Edge; stops only its own server.

Run from the clean root: python -B verification/run_browser_check.py
No dependency installs, model training or external service calls.
"""
from pathlib import Path
import argparse
import json
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--restart-only", action="store_true")
args = parser.parse_args()
evidence = root / "phase2/release/evidence" / datetime.now(timezone.utc).strftime("browser_%Y%m%dT%H%M%S_%fZ")
evidence.mkdir(parents=True, exist_ok=False)
with socket.socket() as reservation:
    reservation.bind(("127.0.0.1", 0))
    port = reservation.getsockname()[1]
log = (evidence / "server.log").open("w", encoding="utf-8")
server = subprocess.Popen([sys.executable, "-B", "-m", "streamlit", "run", "phase2/release/app.py",
                           "--server.address", "127.0.0.1", "--server.port", str(port),
                           "--server.headless", "true", "--browser.gatherUsageStats", "false",
                           "--server.maxUploadSize", "20", "--client.toolbarMode", "minimal"], cwd=root, stdout=log, stderr=subprocess.STDOUT,
                          creationflags=subprocess.CREATE_NO_WINDOW)
result = {}
try:
    deadline = time.monotonic() + 40
    while True:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=1) as response:
                assert response.status == 200
            break
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.2)
    with (evidence / "browser_console.log").open("w", encoding="utf-8") as console:
        child = subprocess.run([sys.executable, "-B", "-m", "phase2.release.browser_smoke",
                                "--evidence-dir", str(evidence), "--port", str(port)] +
                               (["--restart-only"] if args.restart_only else []), cwd=root, stdout=console,
                               stderr=subprocess.STDOUT, timeout=180, creationflags=subprocess.CREATE_NO_WINDOW)
    result["browser_exit_code"] = child.returncode
finally:
    if server.poll() is None:
        subprocess.run(["taskkill", "/PID", str(server.pid), "/T", "/F"], capture_output=True)
    try:
        server.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait(timeout=5)
    log.close()
    result["owned_server_stopped"] = server.poll() is not None
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=1).close()
        result["health_unavailable_after_stop"] = False
    except OSError:
        result["health_unavailable_after_stop"] = True
    result["port"] = port
    (evidence / "lifecycle.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps({**result, "evidence": str(evidence.relative_to(root))}, indent=2))
raise SystemExit(result.get("browser_exit_code", 1))
