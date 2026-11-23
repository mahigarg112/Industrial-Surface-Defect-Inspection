"""Focused standalone checks; no training, selection or full-dataset inference.

Run from this folder: python -B verify_clean.py
Outputs only to a fresh verification/run_* directory. Installed dependencies may
live outside the release; all project imports and model/data reads stay inside.
"""
import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.pop("SDI_PHASE2_OUTPUT", None)
from phase2.common import sha, SmallUNet
from phase2 import data
from phase2.patches import boxes, blend
from phase2.release.inference import (Inspector, MODEL_SHA, PACKAGE, decode_upload,
                                     render, inspection_report, report_bytes)


def main():
    output = ROOT / "verification" / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    checks = []
    details = {}

    def check(name, action):
        try:
            details[name] = action()
            checks.append({"name": name, "passed": True})
            print("PASS", name, flush=True)
        except Exception as exc:
            checks.append({"name": name, "passed": False, "error": repr(exc)})
            print("FAIL", name, repr(exc), flush=True)

    def imports():
        modules = ["common", "model", "dataset_utils", "data", "patches", "train",
                   "objectives", "evaluate", "metrics", "infer", "finalize", "audit"]
        paths = {}
        for name in modules:
            module = importlib.import_module("phase2." + name)
            path = Path(module.__file__).resolve()
            assert path.is_relative_to(ROOT)
            paths[name] = str(path.relative_to(ROOT))
        assert Path(sys.modules[SmallUNet.__module__].__file__).resolve() == ROOT / "phase2/model.py"
        assert PACKAGE.resolve().is_relative_to(ROOT)
        assert not (ROOT / "data").exists(), "This check assumes the dataset has not been linked."
        return paths
    check("standalone_imports_without_dataset", imports)

    image_path = ROOT / "phase2/release/demo_inputs/0007a71bf.jpg"
    image, meta = decode_upload(image_path.read_bytes(), image_path.name)
    cpu = Inspector("cpu")

    def layers():
        rows, handles = [], []
        def hook(name):
            def record(module, inputs, result):
                rows.append({"name": name, "input": list(inputs[0].shape),
                             "operation": repr(module), "output": list(result.shape)})
            return record
        for name, module in cpu.model.named_modules():
            if name and not list(module.children()):
                handles.append(module.register_forward_hook(hook(name)))
        try:
            with torch.inference_mode():
                result = cpu.model(torch.zeros(1, 3, 256, 256))
            assert result.shape == (1, 4, 256, 256)
        finally:
            for handle in handles:
                handle.remove()
        assert sum(p.numel() for p in cpu.model.parameters()) == 482788
        (output / "verified_layers.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        table = ["Layer name | Input shape (N=1) | Operation | Output shape | Purpose",
                 "-----------|-------------------|-----------|--------------|--------"]
        for row in rows:
            op = row["operation"]
            purpose = ("Restore spatial size" if op.startswith("ConvTranspose") else
                       "Learn four logits" if row["name"] == "output" else
                       "Learn local features" if op.startswith("Conv2d") else
                       "Normalize channels" if op.startswith("BatchNorm") else
                       "Nonlinear activation" if op.startswith("ReLU") else "Downsample by two")
            table.append(f"{row['name']} | {row['input']} | {op} | {row['output']} | {purpose}")
        (output / "VERIFIED_LAYER_TABLE.txt").write_text("\n".join(table) + "\n", encoding="utf-8")
        return {"parameters": 482788, "leaf_operations": len(rows), "native_patch_logits": list(result.shape)}
    check("checkpoint_and_all_layer_shapes", layers)

    def preprocessing():
        row = next(r for r in data.records("val") if r["image_id"] == image_path.name)
        with patch.object(data, "IMAGES", image_path.parent):
            original, truth = data.native(row)
            dataset = data.TrainDataset([row], "patch", foreground_crops=True)
            tensor, masks = dataset[0]
        assert tensor.shape == (3, 256, 256) and masks.shape == (4, 256, 256)
        assert tensor.dtype == torch.float32 and 0 <= tensor.min() <= tensor.max() <= 1
        assert set(masks.unique().tolist()) <= {0.0, 1.0}
        np.testing.assert_array_equal(original, image)
        coords = boxes(256, 1600)
        assert len(coords) == 12 and coords[-1] == (0, 1344, 256, 1600)
        crops = [truth[:, y:b, x:r].astype(np.float32) for y, x, b, r in coords]
        np.testing.assert_allclose(blend(crops, coords, 256, 1600), truth, atol=4e-7)
        return {"training_tensor": list(tensor.shape), "native_truth": list(truth.shape),
                "patch_count": len(coords), "no_parent_data_access": True}
    check("real_sample_preprocessing_and_reconstruction", preprocessing)

    def inference(device):
        inspector = cpu if device == "cpu" else Inspector(device)
        captured = []
        def capture(module, inputs):
            if not captured:
                captured.append(inputs[0][0].detach().cpu().numpy().copy())
        handle = inspector.model.register_forward_pre_hook(capture)
        try:
            probability, timing = inspector.probabilities(image)
        finally:
            handle.remove()
        expected = image[:, :256].transpose(2, 0, 1).astype(np.float32) / 255
        # CUDA float32 division may differ from CPU/NumPy by one rounding unit.
        np.testing.assert_allclose(captured[0], expected, rtol=0, atol=1e-7)
        assert probability.shape == (4, 256, 1600)
        assert np.isfinite(probability).all() and 0 <= probability.min() <= probability.max() <= 1
        calls = inspector.forward_runs
        view = render(image, probability, inspector.locked_thresholds)
        changed = render(image, probability, (1, 1, 1, 1))
        assert inspector.forward_runs == calls
        assert not np.array_equal(view["masks"], changed["masks"])
        if device == "cpu":
            (output / "overlay.png").write_bytes(view["overlay_png"])
            for c, payload in enumerate(view["mask_pngs"], 1):
                (output / f"class_{c}.png").write_bytes(payload)
                np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(payload))) > 0, view["masks"][c-1])
            report = inspection_report(meta, timing, view, inspector.locked_thresholds)
            (output / "inspection.json").write_bytes(report_bytes(report))
            assert json.loads(report_bytes(report))["operating_point"] == "locked_validation_settings"
        return {"shape": list(probability.shape), "mask_pixels": view["positive_pixels"],
                "timing": timing, "threshold_change_without_forward": True}
    check("actual_cpu_inference_and_exports", lambda: inference("cpu"))
    if torch.cuda.is_available():
        check("actual_cuda_inference", lambda: inference("cuda"))
    else:
        details["cuda"] = "Unavailable; CPU passed separately."

    def saved_arithmetic():
        from phase2 import verify_evidence
        # Preserve the copied historical verification report; route this invocation to fresh evidence.
        with patch.object(verify_evidence, "write_json", side_effect=lambda path, value:
                          (output / "saved_evidence_check.json").write_text(json.dumps(value, indent=2), encoding="utf-8")), \
             contextlib.redirect_stdout(io.StringIO()):
            return verify_evidence.verify()
    check("saved_metrics_and_locked_snapshot_arithmetic", saved_arithmetic)

    def startup():
        import urllib.request
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        cmd = [sys.executable, "-B", "-m", "streamlit", "run", "phase2/release/app.py",
               "--server.address", "127.0.0.1", "--server.port", str(port),
               "--server.headless", "true", "--browser.gatherUsageStats", "false"]
        log = (output / "streamlit_startup.log").open("w", encoding="utf-8")
        process = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            deadline = time.monotonic() + 40
            while True:
                if process.poll() is not None:
                    raise RuntimeError("Streamlit exited before health check")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=1) as response:
                        assert response.status == 200
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.2)
            with urllib.request.urlopen(f"http://127.0.0.1:{port}", timeout=3) as response:
                assert response.status == 200
        finally:
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            log.close()
        return {"health_http": 200, "page_http": 200, "port": port, "owned_process_stopped": process.poll() is not None,
                "scope": "HTTP startup; actual widget behavior is tested separately with Streamlit AppTest/browser."}
    check("streamlit_http_startup_and_teardown", startup)

    assert sha(PACKAGE / "model.pt") == MODEL_SHA
    summary = {"passed": all(item["passed"] for item in checks), "checks": checks, "details": details,
               "scope": "No retraining or full-dataset inference. Existing metrics checked arithmetically only.",
               "python": sys.version, "torch": torch.__version__, "output": str(output.relative_to(ROOT))}
    (output / "integration.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (ROOT / "verification/latest_integration.txt").write_text(str(output.relative_to(ROOT)), encoding="utf-8")
    print(json.dumps({"passed": summary["passed"], "checks": len(checks), "output": str(output)}, indent=2))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
