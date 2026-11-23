"""Geometry, RLE, four-class arithmetic and inference regression tests."""
import unittest

import numpy as np
import torch

from .common import decode_rle
from .metrics import counts, summarize
from .patches import blend, boxes, predict, starts


class PipelineTests(unittest.TestCase):
    def test_rle_known_column_major_and_invalid(self):
        mask = decode_rle("1 1 3 2 15 1", height=3, width=5)
        expected = np.zeros((3, 5), np.uint8)
        expected[0,0] = expected[2,0] = expected[0,1] = expected[2,4] = 1
        np.testing.assert_array_equal(mask, expected)
        for rle in ["1", "0 1", "15 2", "1 -1"]:
            with self.assertRaises(ValueError):
                decode_rle(rle, height=3, width=5)
        self.assertEqual(decode_rle("", 3, 5).sum(), 0)

    def test_rle_independent_roundtrip_four_channels(self):
        rng = np.random.default_rng(42)
        for _ in range(4):
            mask = rng.integers(0, 2, (19, 31), dtype=np.uint8)
            flat = mask.T.reshape(-1)
            edges = np.flatnonzero(np.diff(np.pad(flat.astype(int), (1,1))))
            pairs = np.stack([edges[::2]+1, edges[1::2]-edges[::2]], axis=1)
            encoded = " ".join(map(str, pairs.reshape(-1)))
            np.testing.assert_array_equal(decode_rle(encoded, 19, 31), mask)

    def test_native_crop_reconstruction_alignment_edges(self):
        rng = np.random.default_rng(27)
        original = rng.random((4, 256, 1600), dtype=np.float32)
        coordinates = boxes(256, 1600)
        self.assertEqual(coordinates[-1], (0, 1344, 256, 1600))
        for kind in ["hann", "uniform"]:
            crops = [original[:,y:b,x:r] for y,x,b,r in coordinates]
            rebuilt = blend(crops, coordinates, 256, 1600, kind=kind)
            np.testing.assert_allclose(rebuilt, original, atol=4e-7)
        with self.assertRaises(ValueError):
            blend(crops[:-1], coordinates[:-1], 256, 1600)

    def test_small_odd_images_coverage(self):
        for height, width in [(17,23), (257,1601), (256,1600)]:
            coordinates = boxes(height, width)
            crops = [np.full((4,256,256), 0.37, np.float32) for _ in coordinates]
            np.testing.assert_allclose(blend(crops, coordinates, height, width), 0.37, atol=1e-7)
        with self.assertRaises(ValueError):
            starts(1600, 256, 257)

    def test_four_class_metrics_and_large_int64(self):
        totals = np.array([[2,3,0,7],[1,0,4,2],[1,1,5,3]], dtype=np.int64)
        result = summarize(totals)
        self.assertAlmostEqual(result["per_class"]["1"]["dice"], 2/3)
        self.assertEqual(result["per_class"]["3"]["dice"], 0)
        self.assertAlmostEqual(result["per_class"]["4"]["recall"], 0.7)
        totals *= 100000000
        self.assertAlmostEqual(summarize(totals)["macro_dice"], result["macro_dice"])
        target = np.eye(4, dtype=bool)[:,:,None]
        np.testing.assert_array_equal(counts(target,target), [[1]*4,[0]*4,[0]*4])
        self.assertEqual(summarize(counts(target,target))["macro_dice"], 1)
        self.assertIsNone(summarize(np.zeros((3,4),dtype=np.int64))["macro_dice"])

    def test_batched_prediction_coordinates(self):
        class Pointwise(torch.nn.Module):
            def forward(self, x):
                return torch.cat([x, x[:, :1]], dim=1)
        torch.set_num_threads(2)
        image = np.random.default_rng(1).integers(0,256,(73,611,3),dtype=np.uint8)
        reference = torch.from_numpy(np.concatenate([image.transpose(2,0,1),image[None,:,:,0]],0).copy()).float().div(255).sigmoid().numpy()
        for batch in [1,3,8]:
            actual = predict(Pointwise(), image, "patch", device="cpu", batch_size=batch)
            np.testing.assert_allclose(actual,reference,atol=4e-7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
