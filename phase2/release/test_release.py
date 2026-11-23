"""Small release tests. Real model smokes are separate from research evaluation."""
import io
import json
import socket
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image, ImageOps
import torch
from streamlit.testing.v1 import AppTest

from .inference import (Inspector, LOCKED_THRESHOLDS, MAX_BYTES, decode_upload,
                        render, inspection_report, report_bytes)

HERE = Path(__file__).parent


def png(width=48, height=32):
    output = io.BytesIO()
    Image.new("RGB", (width, height), (70, 90, 120)).save(output, "PNG")
    return output.getvalue()


class BackendTests(unittest.TestCase):
    def test_exif_orientation_and_empty_overlapping_masks(self):
        image = Image.new("RGB", (48, 32), (30, 80, 120))
        exif = image.getexif(); exif[274] = 6
        content = io.BytesIO(); image.save(content, "JPEG", exif=exif)
        decoded, meta = decode_upload(content.getvalue(), "oriented.jpg")
        expected = np.asarray(ImageOps.exif_transpose(Image.open(io.BytesIO(content.getvalue()))).convert("RGB"))
        np.testing.assert_array_equal(decoded, expected)
        self.assertEqual(decoded.shape, (48, 32, 3))
        self.assertTrue(meta["exif_orientation_applied"])
        p = np.zeros((4, 48, 32), np.float32)
        empty = render(decoded, p, LOCKED_THRESHOLDS)
        np.testing.assert_array_equal(empty["overlay"], decoded)
        self.assertEqual(empty["positive_pixels"], [0]*4)
        p[:2, 3, 5] = 1
        overlap = render(decoded, p, LOCKED_THRESHOLDS)
        self.assertEqual(overlap["positive_pixels"], [1, 1, 0, 0])
        from .inference import COLORS, OVERLAY_ALPHA
        expected_color = ((1-OVERLAY_ALPHA)*decoded[3,5] + OVERLAY_ALPHA*COLORS[:2].mean(0)).astype(np.uint8)
        np.testing.assert_array_equal(overlap["overlay"][3,5], expected_color)

    def test_upload_decoding_and_rejections(self):
        image, meta = decode_upload(png(), "sample.PNG")
        self.assertEqual(image.shape, (32, 48, 3))
        self.assertEqual((meta["width"], meta["height"]), (48, 32))
        for data, name in [(b"", "empty.jpg"), (b"not an image", "x.jpg"),
                           (png(), "x.svg"), (png(), "wrong.jpg"),
                           (b"x" * (MAX_BYTES + 1), "large.png"), (png(8193, 16), "wide.png")]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                decode_upload(data, name)

    def test_native_alignment_exports_and_thresholds(self):
        image, meta = decode_upload(png(), "sample.png")
        probability = np.zeros((4, 32, 48), np.float32)
        for c in range(4):
            probability[c, 3+c:5+c, 7+c:10+c] = 1
        view = render(image, probability, LOCKED_THRESHOLDS)
        np.testing.assert_array_equal(view["masks"].sum((1, 2)), [6] * 4)
        np.testing.assert_array_equal(view["overlay"][0, 0], image[0, 0])
        for c in range(4):
            np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(view["mask_pngs"][c]))) > 0,
                                          probability[c] > 0)
        self.assertEqual(Image.open(io.BytesIO(view["overlay_png"])).size, (48, 32))
        report = json.loads(report_bytes(inspection_report(meta, {"device": "cpu"}, view, LOCKED_THRESHOLDS)))
        self.assertEqual(report["operating_point"], "locked_validation_settings")
        self.assertIsNone(report["accuracy_on_this_image"])
        self.assertEqual([v["predicted_pixels"] for v in report["classes"]], [6] * 4)
        changed = inspection_report(meta, {}, view, [0.1] * 4)
        self.assertEqual(changed["operating_point"], "exploratory_user_settings_not_evaluated")
        for thresholds in [[0.5]*3, [float("nan")]*4, [-0.1]*4, [1.1]*4]:
            with self.assertRaises(ValueError):
                render(image, probability, thresholds)

    def test_wrong_checkpoint_rejected(self):
        with mock.patch("phase2.release.inference.sha", return_value="incorrect"):
            with self.assertRaises(ValueError):
                Inspector("cpu")

    def test_offline_cpu_and_auto_fallback(self):
        with mock.patch.object(socket, "create_connection", side_effect=AssertionError("Network forbidden")), \
             mock.patch("urllib.request.urlopen", side_effect=AssertionError("Network forbidden")), \
             mock.patch("torch.cuda.is_available", return_value=False):
            inspector = Inspector("auto")
            image, _ = decode_upload(png(), "sample.png")
            probability, timing = inspector.probabilities(image)
        self.assertEqual(probability.shape, (4, 32, 48))
        self.assertEqual(timing["device"], "cpu")
        self.assertTrue(timing["fallback_reason"])

    def test_simulated_gpu_memory_fallback(self):
        inspector = Inspector("cpu")
        inspector.preference, inspector.device = "auto", "cuda"
        image, _ = decode_upload(png(), "sample.png")
        with mock.patch("phase2.release.inference.predict", side_effect=[torch.OutOfMemoryError("simulated"), np.zeros((4,32,48),np.float32)]), \
             mock.patch("torch.cuda.synchronize"), mock.patch("torch.cuda.empty_cache"):
            _, timing = inspector.probabilities(image)
        self.assertEqual(timing["device"], "cpu")
        self.assertIn("memory", timing["fallback_reason"])


class AppTests(unittest.TestCase):
    def test_inspect_threshold_cache_downloads_and_new_sample(self):
        app = AppTest.from_file(HERE / "app.py", default_timeout=90).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        app.selectbox(key="sample").set_value("0007a71bf.jpg").run()
        app.button(key="inspect").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        self.assertEqual(len(app.metric), 4)
        self.assertEqual(len(app.get("download_button")), 6)
        self.assertEqual(app.session_state["network_runs"], 1)
        original_area = app.metric[2].value
        app.toggle(key="adjust").set_value(True).run()
        app.slider(key="threshold_3").set_value(0.99).run()
        self.assertEqual(app.session_state["network_runs"], 1)
        self.assertNotEqual(app.metric[2].value, original_area)
        app.toggle(key="adjust").set_value(False).run()
        self.assertEqual(app.metric[2].value, original_area)
        app.selectbox(key="sample").set_value("000ccc2ac.jpg").run()
        self.assertNotIn("inspection", app.session_state)
        app.button(key="inspect").click().run()
        self.assertEqual(app.session_state["network_runs"], 2)
        self.assertEqual(len(app.error), 0)

    def test_empty_prediction_is_neutral_and_advanced_section_collapsed(self):
        app = AppTest.from_file(HERE / "app.py", default_timeout=90).run()
        self.assertEqual(len(app.sidebar), 0)
        self.assertTrue(any("Research Prototype" in x.value and "Not Validated for Industrial Use" in x.value for x in app.info))
        advanced = next(x for x in app.expander if x.label == "Advanced: Explore Prediction Thresholds")
        self.assertFalse(advanced.proto.expanded)
        app.selectbox(key="sample").set_value("0007a71bf.jpg").run()
        def empty_probability(image):
            return np.zeros((4, *image.shape[:2]), np.float32), {
                "inference_seconds": .001, "device": "cpu", "device_name": "CPU", "fallback_reason": None}
        with mock.patch("phase2.release.inference.Inspector.probabilities", side_effect=empty_probability):
            app.button(key="inspect").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("No defect predicted" in x.value and "does not guarantee" in x.value for x in app.markdown))
        self.assertTrue(all(x.value == "0.000%" for x in app.metric))

    def test_invalid_input_and_inference_exception_are_visible(self):
        app = AppTest.from_file(HERE / "app.py", default_timeout=90)
        app.session_state["sample_input"] = (b"", "empty.jpg")
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertIn("empty", app.error[0].value.lower())
        app.selectbox(key="sample").set_value("0007a71bf.jpg").run()
        with mock.patch("phase2.release.inference.Inspector.probabilities", side_effect=RuntimeError("simulated inference error")):
            app.button(key="inspect").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertIn("Inference failed", app.error[0].value)
        self.assertNotIn("inspection", app.session_state)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__, fromlist=["*"])))
    output = HERE / "evidence"
    output.mkdir(exist_ok=True)
    (output / "release_tests.json").write_text(json.dumps({"passed":result.wasSuccessful(),"tests_run":result.testsRun,
        "failures":[name for name,_ in result.failures],"errors":[name for name,_ in result.errors],
        "scope":"backend validation/exports/offline/CPU fallback plus real Streamlit AppTest inspection/cache/error paths; simulated OOM explicitly labeled"},indent=2),encoding="utf-8")
    raise SystemExit(0 if result.wasSuccessful() else 1)
