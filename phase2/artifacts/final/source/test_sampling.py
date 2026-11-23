import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .common import OUT,write_json
from .data import TrainDataset


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((4,256,1600),np.uint8)
        self.mask[0,99,1490] = self.mask[1,100,1500] = 1
        self.image = np.zeros((256,1600,3),np.uint8)
        self.image[:,:,0] = self.mask[1]*255
        self.image[:,:,1] = self.mask[0]*255
        self.image[:,:,2] = np.arange(1600,dtype=np.uint16)[None,:]%251
        self.rows = [{"rle":"1 1|2 1||"},{"rle":"1 1|||"}]

    def test_rare_class_crop_preserves_all_channels_and_alignment(self):
        dataset = TrainDataset(self.rows,"patch",foreground_crops=True)
        with patch("phase2.data.native",return_value=(self.image,self.mask)):
            image,mask = dataset[0]
            image_again,mask_again = dataset[0]
        self.assertEqual(float(mask[1].sum()),1)
        self.assertEqual(float(mask[0].sum()),1)
        self.assertTrue(torch.equal(image[0],mask[1]))
        self.assertTrue(torch.equal(image[1],mask[0]))
        self.assertTrue(torch.equal(image,image_again) and torch.equal(mask,mask_again))
        self.assertEqual(dataset.class_counts,[2,1,0,0])

    def test_uniform_default_matches_original_index_rule(self):
        dataset = TrainDataset(self.rows,"patch",seed=42,epoch=2)
        rng = np.random.default_rng([42,2,0])
        y,x,b,r = dataset.coordinates[int(rng.integers(len(dataset.coordinates)))]
        with patch("phase2.data.native",return_value=(self.image,self.mask)):
            image,mask = dataset[0]
        np.testing.assert_array_equal(mask.numpy(),self.mask[:,y:b,x:r])
        np.testing.assert_allclose(image.numpy(),self.image[y:b,x:r].transpose(2,0,1)/255,atol=1e-7,rtol=0)

    def test_negative_images_keep_uniform_sampling_and_whole_rejected(self):
        mask = np.zeros_like(self.mask)
        rows = [{"rle":"|||"}]
        uniform = TrainDataset(rows,"patch")
        focused = TrainDataset(rows,"patch",foreground_crops=True)
        with patch("phase2.data.native",return_value=(self.image,mask)):
            a,b = uniform[0]
            c,d = focused[0]
        self.assertTrue(torch.equal(a,c) and torch.equal(b,d))
        with self.assertRaises(ValueError):
            TrainDataset(rows,"whole",foreground_crops=True)


if __name__ == "__main__":
    result = unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SamplingTests))
    write_json(OUT/"foreground_sampling_tests.json",{"passed":result.wasSuccessful(),"tests_run":result.testsRun})
    sys.exit(0 if result.wasSuccessful() else 1)
