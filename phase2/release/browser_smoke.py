"""Local Edge/CDP smoke with real upload/download; no browser installation."""
import argparse
import base64
import io
import json
import socket
import subprocess
import tempfile
import time
import urllib.request
from contextlib import ExitStack
from pathlib import Path

from PIL import Image
from websockets.sync.client import connect

HERE = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restart-only", action="store_true")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--evidence-dir", type=Path, default=HERE / "evidence",
                        help="Fresh verification output folder; existing records are never overwritten")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError("Invalid loopback port")
    base_url = f"http://127.0.0.1:{args.port}"
    evidence = args.evidence_dir.resolve()
    if not evidence.is_relative_to((HERE / "evidence").resolve()):
        raise ValueError("Browser evidence must remain inside this release's evidence directory")
    evidence.mkdir(parents=True, exist_ok=True)
    name = "browser_restart" if args.restart_only else "browser_smoke"
    target = evidence / f"{name}.json"
    if target.exists():
        raise FileExistsError("Preserve the existing browser test record")
    with urllib.request.urlopen(base_url + "/_stcore/health", timeout=5) as response:
        assert response.status == 200
    executable = Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")
    if not executable.exists():
        raise FileNotFoundError("This optional browser test requires installed Microsoft Edge.")
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        debug_port = reservation.getsockname()[1]
    checks = {"server_health": True}
    with tempfile.TemporaryDirectory(prefix="browser-profile-", dir=evidence, ignore_cleanup_errors=True) as temporary:
        # Check before the TemporaryDirectory context's recursive cleanup.
        assert Path(temporary).resolve().parent == evidence.resolve()
        assert Path(temporary).name.startswith("browser-profile-")
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = subprocess.SW_HIDE
        driver_log = (evidence / "browser_driver.log").open("ab")
        connections = ExitStack()
        process = subprocess.Popen([str(executable), "--headless=new", "--disable-gpu", f"--remote-debugging-port={debug_port}",
                                    "--remote-debugging-address=127.0.0.1", "--remote-allow-origins=http://127.0.0.1",
                                    f"--user-data-dir={temporary}", "--no-first-run", "--disable-background-networking",
                                    "--disable-component-update", "--disable-sync", "--no-default-browser-check", "about:blank"],
                                   startupinfo=startup, stdout=subprocess.DEVNULL, stderr=driver_log)
        ws = None
        try:
            deadline = time.monotonic() + 30
            while True:
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{debug_port}/json", timeout=2) as response:
                        pages = json.load(response)
                    page = next(page for page in pages if page["type"] == "page")
                    break
                except Exception:
                    if time.monotonic() > deadline:
                        raise RuntimeError("Edge debugger did not become ready")
                    time.sleep(.2)
            ws = connections.enter_context(connect(page["webSocketDebuggerUrl"], origin="http://127.0.0.1", max_size=16*1024*1024))
            sequence = 0
            download_events = []

            def call(method, params=None):
                nonlocal sequence
                sequence += 1
                request_id = sequence
                ws.send(json.dumps({"id":request_id,"method":method,"params":params or {}}))
                while True:
                    message = json.loads(ws.recv(timeout=40))
                    event = message.get("method", "")
                    params = message.get("params", {})
                    if event.startswith("Browser.download") or event.startswith("Page.download"):
                        download_events.append({"event": event, "data": params})
                    elif event == "Network.requestWillBeSent" and "/media/" in params.get("request", {}).get("url", ""):
                        download_events.append({"event": event, "url": params["request"]["url"]})
                    elif event == "Network.responseReceived" and "/media/" in params.get("response", {}).get("url", ""):
                        download_events.append({"event": event, "url": params["response"]["url"], "status": params["response"]["status"]})
                    elif event == "Network.loadingFailed":
                        download_events.append({"event": event, "error": params.get("errorText"), "blockedReason": params.get("blockedReason")})
                    if message.get("id") == request_id:
                        if "error" in message:
                            raise RuntimeError(message["error"])
                        return message.get("result", {})

            def evaluate(expression):
                response = call("Runtime.evaluate", {"expression":expression,"returnByValue":True})
                if "exceptionDetails" in response:
                    raise RuntimeError(response["exceptionDetails"])
                return response.get("result", {}).get("value")

            def wait(expression, seconds=60):
                deadline = time.monotonic() + seconds
                while time.monotonic() < deadline:
                    if evaluate(expression):
                        return
                    time.sleep(.2)
                raise AssertionError(f"Browser condition timed out: {expression}")

            def upload(path):
                document = call("DOM.getDocument")
                node = call("DOM.querySelector", {"nodeId":document["root"]["nodeId"],"selector":"input[type=file]"})
                assert node["nodeId"]
                call("DOM.setFileInputFiles", {"nodeId":node["nodeId"],"files":[str(path.resolve())]})

            call("Page.enable")
            call("Network.enable")
            # The app uses HTTP loopback; block HTTPS to verify it needs no remote assets/API.
            call("Network.setBlockedURLs", {"urls":["https://*"]})
            call("Emulation.setDeviceMetricsOverride", {"width":1440,"height":1600,"deviceScaleFactor":1,"mobile":False})
            call("Page.navigate", {"url":base_url})
            wait("document.querySelector('input[type=file]') !== null")
            wait("document.body.innerText.includes('Research Prototype')")
            assert "Steel surface inspection" in evaluate("document.body.innerText")
            checks["app_loaded_and_checkpoint_ready"] = True
            checks["external_https_blocked"] = True
            if not args.restart_only:
                upload(HERE / "demo_inputs/0007a71bf.jpg")
                wait("document.body.innerText.includes('Original image')")
                wait("[...document.querySelectorAll('button')].some(x=>x.innerText.trim()==='Inspect Image')")
                evaluate("[...document.querySelectorAll('button')].find(x=>x.innerText.trim()==='Inspect Image').click()")
                wait("document.body.innerText.includes('Processing time:')")
                wait("document.querySelectorAll('[data-testid=stMetricValue]').length === 4")
                checks["real_upload_inference_four_masks"] = True
                screenshot = call("Page.captureScreenshot", {"format":"png","captureBeyondViewport":True})
                (evidence/"application_locked.png").write_bytes(base64.b64decode(screenshot["data"]))
                # Toggle and adjust an actual browser slider; AppTest separately checks no network rerun.
                evaluate("[...document.querySelectorAll('[data-testid=stExpander]')].find(x=>x.querySelector('summary')?.innerText.includes('Advanced: Explore Prediction Thresholds')).querySelector('summary').click()")
                evaluate("document.querySelector('input[aria-label=\"Adjust thresholds\"]').focus()")
                call("Input.dispatchKeyEvent", {"type":"keyDown","key":" ","code":"Space","windowsVirtualKeyCode":32})
                call("Input.dispatchKeyEvent", {"type":"keyUp","key":" ","code":"Space","windowsVirtualKeyCode":32})
                wait("document.querySelectorAll('[role=slider],input[type=range]').length === 4")
                area_before = evaluate("document.querySelectorAll('[data-testid=stMetricValue]')[2].innerText")
                before = evaluate("document.querySelectorAll('[role=slider],input[type=range]')[2].getAttribute('aria-valuenow') || document.querySelectorAll('[role=slider],input[type=range]')[2].value")
                evaluate("document.querySelectorAll('[role=slider],input[type=range]')[2].focus()")
                call("Input.dispatchKeyEvent", {"type":"keyDown","key":"ArrowRight","code":"ArrowRight","windowsVirtualKeyCode":39})
                call("Input.dispatchKeyEvent", {"type":"keyUp","key":"ArrowRight","code":"ArrowRight","windowsVirtualKeyCode":39})
                wait(f"(document.querySelectorAll('[role=slider],input[type=range]')[2].getAttribute('aria-valuenow') || document.querySelectorAll('[role=slider],input[type=range]')[2].value) !== {json.dumps(before)}")
                # The range input changes before Python emits the new masks and
                # download URLs. Wait for the actual result, not just the thumb.
                wait(f"document.querySelectorAll('[data-testid=stMetricValue]')[2].innerText !== {json.dumps(area_before)}")
                wait("![...document.querySelectorAll('[data-testid=stStatusWidget]')].some(x=>x.innerText.includes('Running'))")
                time.sleep(.5)
                downloads = evidence / "browser_downloads"
                downloads.mkdir(exist_ok=True)
                call("Browser.setDownloadBehavior", {"behavior":"allow","downloadPath":str(downloads.resolve()),"eventsEnabled":True})
                evaluate("[...document.querySelectorAll('[data-testid=stExpander]')].find(x=>x.querySelector('summary')?.innerText.includes('Download individual masks')).querySelector('summary').click()")
                requested_downloads = [("Download overlay PNG", "steel_overlay.png"),
                                       ("Download inspection JSON", "steel_inspection.json")]
                requested_downloads += [(f"Download Class {c} mask PNG", f"steel_class_{c}.png") for c in range(1, 5)]
                for label, filename in requested_downloads:
                    wait(f"[...document.querySelectorAll('button')].some(x=>x.innerText.trim()==={json.dumps(label)} && !x.disabled)")
                    point = evaluate(f"(()=>{{const b=[...document.querySelectorAll('button')].find(x=>x.innerText.trim()==={json.dumps(label)}); b.scrollIntoView({{block:'center'}}); const r=b.getBoundingClientRect(); return {{x:r.x+r.width/2,y:r.y+r.height/2}};}})()")
                    call("Input.dispatchMouseEvent", {"type":"mousePressed","x":point["x"],"y":point["y"],"button":"left","clickCount":1})
                    call("Input.dispatchMouseEvent", {"type":"mouseReleased","x":point["x"],"y":point["y"],"button":"left","clickCount":1})
                    deadline = time.monotonic()+15
                    while not (downloads/filename).exists() and time.monotonic()<deadline:
                        evaluate("document.readyState")
                        time.sleep(.1)
                    (evidence / "download_events.json").write_text(json.dumps(download_events, indent=2), encoding="utf-8")
                    assert (downloads/filename).is_file(), filename
                with Image.open(downloads/"steel_overlay.png") as overlay:
                    overlay.load()
                    assert overlay.size == (1600,256)
                for c in range(1, 5):
                    with Image.open(downloads / f"steel_class_{c}.png") as mask:
                        mask.load()
                        assert mask.size == (1600, 256) and mask.mode == "L"
                        assert {value for _, value in mask.getcolors(256)} <= {0, 255}
                report = json.loads((downloads/"steel_inspection.json").read_text())
                assert len(report["classes"]) == 4 and report["input"]["width"] == 1600
                assert report["operating_point"] == "exploratory_user_settings_not_evaluated"
                assert report["thresholds"][2] != .5
                checks["threshold_control_and_real_png_json_downloads"] = True
                checks["four_individual_binary_png_downloads"] = True
                screenshot = call("Page.captureScreenshot", {"format":"png","captureBeyondViewport":True})
                (evidence/"application.png").write_bytes(base64.b64decode(screenshot["data"]))
                evaluate("document.querySelector('input[aria-label=\"Adjust thresholds\"]').focus()")
                call("Input.dispatchKeyEvent", {"type":"keyDown","key":" ","code":"Space","windowsVirtualKeyCode":32})
                call("Input.dispatchKeyEvent", {"type":"keyUp","key":" ","code":"Space","windowsVirtualKeyCode":32})
                wait("document.querySelectorAll('[role=slider],input[type=range]').length === 0")
                upload(HERE / "demo_inputs/000ccc2ac.jpg")
                wait("document.body.innerText.includes('Click Inspect Image to generate predictions.')")
                evaluate("[...document.querySelectorAll('button')].find(x=>x.innerText.trim()==='Inspect Image').click()")
                wait("document.body.innerText.includes('Processing time:')")
                wait("document.querySelectorAll('[data-testid=stMetricValue]').length === 4")
                screenshot = call("Page.captureScreenshot", {"format":"png","captureBeyondViewport":True})
                (evidence/"application_second.png").write_bytes(base64.b64decode(screenshot["data"]))
                checks["second_authorized_image_real_inference"] = True
                for filename, data, expected in [("invalid.jpg",b"not an image","cannot be decoded"),("empty.jpg",b"","empty")]:
                    path = downloads/filename
                    path.write_bytes(data)
                    upload(path)
                    wait(f"document.body.innerText.toLowerCase().includes({json.dumps(expected)})")
                checks["real_invalid_and_empty_uploads_handled"] = True
            call("Browser.close")
            checks["passed"] = True
            target.write_text(json.dumps(checks,indent=2),encoding="utf-8")
            print(json.dumps(checks,indent=2))
        except Exception as exc:
            if ws:
                try:
                    debug = evaluate("({text:document.body.innerText, inputs:[...document.querySelectorAll('input')].map(x=>x.outerHTML), sliders:[...document.querySelectorAll('[role=slider]')].map(x=>x.outerHTML)})")
                    debug["failure"] = str(exc)
                    (evidence/f"browser_failure_{len(list(evidence.glob('browser_failure_*.json')))+1}.json").write_text(json.dumps(debug,indent=2),encoding="utf-8")
                except Exception:
                    pass
            raise
        finally:
            if ws:
                connections.close()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Stop only the process tree created above with a private profile.
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, timeout=10)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            driver_log.close()


if __name__ == "__main__":
    main()
