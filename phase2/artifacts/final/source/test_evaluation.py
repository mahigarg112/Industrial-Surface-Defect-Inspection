import json
import tempfile
import unittest
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .common import OUT, write_json
from .evaluate import THRESHOLDS, size_stats, tune
from .metrics import counts, summarize
from .objectives import weighted_loss,present_dice_loss,balanced_present_dice_loss
from .smoke import loss_fn


class EvaluationTests(unittest.TestCase):
    def test_class_balanced_dice_equalizes_positive_pair_frequencies(self):
        logits = torch.zeros(4,4,8,8,requires_grad=True)
        target = torch.zeros_like(logits)
        for c,n in enumerate([4,1,2,1]):
            target[:n,c,:4,:4] = 1
        loss = balanced_present_dice_loss(logits,target,torch.tensor([1,4,2,4]))
        # Each present mask has area 16, prediction area 32 and intersection 8.
        expected = 0.5*np.log(2)+0.5*(1-17/49)
        self.assertAlmostEqual(float(loss.detach()),expected,places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        empty = torch.zeros_like(target)
        self.assertAlmostEqual(float(balanced_present_dice_loss(logits,empty,torch.tensor([1,4,2,4])).detach()),0.5*np.log(2),places=6)

    def test_present_dice_empty_targets_use_bce_and_positive_pairs_match(self):
        logits = torch.zeros(2,4,16,24,requires_grad=True)
        empty = torch.zeros_like(logits)
        loss = present_dice_loss(logits,empty)
        self.assertAlmostEqual(float(loss.detach()),0.5*float(torch.nn.functional.binary_cross_entropy_with_logits(logits,empty).detach()))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        positive = torch.ones_like(logits)
        self.assertAlmostEqual(float(present_dice_loss(logits,positive).detach()),float(loss_fn(logits,positive).detach()))

    def test_weighted_bce_changes_only_positive_weighting(self):
        torch.manual_seed(2)
        logits = torch.randn(2,4,16,24,requires_grad=True)
        truth = (torch.rand(2,4,16,24)>0.9).float()
        self.assertEqual(float(weighted_loss(logits,truth,torch.ones(4)).detach()),float(loss_fn(logits,truth).detach()))
        weighted = weighted_loss(logits,truth,torch.tensor([1000,1000,40,180]))
        self.assertTrue(torch.isfinite(weighted))
        weighted.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_threshold_histogram_matches_brute_force_including_boundaries(self):
        values = np.array([0,0.00001,0.01,0.1,0.5,0.9,1],np.float32)
        probability = np.tile(values,(4,3,1))
        truth = np.zeros_like(probability,dtype=np.uint8)
        for c in range(4):
            truth[c,:,c+1:c+3] = 1
        expected = []
        for c in range(4):
            options = []
            for threshold in THRESHOLDS:
                score = summarize(counts(probability>=threshold,truth))["per_class"][str(c+1)]["dice"]
                options.append((score,float(threshold)))
            expected.append(max(options)[1])
        with tempfile.TemporaryDirectory(dir=OUT) as directory:
            path = Path(directory)/"thresholds.json"
            with patch("phase2.evaluate.native",return_value=(np.zeros((3,7,3),np.uint8),truth)), patch("phase2.evaluate.predict",return_value=probability):
                selected = tune(None,"patch",[{}],path)
            self.assertEqual(selected,expected)
            result = json.loads(path.read_text())
            brute = summarize(counts(probability>=np.array(selected)[:,None,None],truth))
            self.assertEqual(result["validation_metrics"],brute)

    def test_component_buckets_and_detection_cutoff(self):
        truth = np.zeros((4,200,240),np.uint8)
        truth[0,0:10,0:10] = 1       # small: 100 pixels
        truth[0,20:60,0:40] = 1     # medium: 1600 pixels
        truth[0,80:180,0:100] = 1   # large: 10000 pixels
        prediction = np.zeros_like(truth,dtype=bool)
        prediction[0,0,0:9] = 1    # 9%: not detected
        prediction[0,20:24,0:40] = 1 # 10%: detected
        prediction[0,80:90,0:100] = 1 # 10%: detected
        result = size_stats(prediction,truth)
        np.testing.assert_array_equal(result["small"][0],[1,0,100,9])
        np.testing.assert_array_equal(result["medium"][0],[1,1,1600,160])
        np.testing.assert_array_equal(result["large"][0],[1,1,10000,1000])
        self.assertEqual(result["small"][1:].sum(),0)


if __name__ == "__main__":
    result = unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(EvaluationTests))
    write_json(OUT/"evaluation_tests.json",{"passed":result.wasSuccessful(),"tests_run":result.testsRun})
    sys.exit(0 if result.wasSuccessful() else 1)
