import unittest
import numpy as np
import torch
from unittest.mock import patch
from phase2.augmentation_study.augmentation import parameters, photometric, AugmentedDataset
from phase2.data import TrainDataset, records


class AugmentationTests(unittest.TestCase):
    def test_identity_and_shape(self):
        x = torch.linspace(0, 1, 3*32*48).reshape(3,32,48)
        self.assertTrue(torch.equal(photometric(x), x))
        self.assertEqual(photometric(x, .9, 1.1).shape, x.shape)

    def test_declared_pixel_formula_no_spatial_change(self):
        x = torch.linspace(.2, .8, 3*24*31).reshape(3,24,31)
        actual = photometric(x, 1.05, .95)
        expected = ((x-x.mean())*.95+x.mean())*1.05
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)

    def test_bounded_finite_rgb_and_deterministic_policy(self):
        x = torch.tensor([0., .2, .5, .8, 1.]).repeat(3,4,1)
        for index in range(100):
            a,b,c = parameters(42,9,index)
            self.assertEqual((a,b,c), parameters(42,9,index))
            self.assertTrue(.9 <= b <= 1.1 and .9 <= c <= 1.1)
            y = photometric(x,b,c)
            self.assertTrue(torch.isfinite(y).all() and y.min() >= 0 and y.max() <= 1)

    def test_augmentation_does_not_consume_training_rng(self):
        state = torch.get_rng_state().clone(); old = np.random.get_state()
        for i in range(20): parameters(42,9,i)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        np.testing.assert_array_equal(old[1], np.random.get_state()[1])

    def test_synthetic_crop_and_mask_alignment_binary(self):
        rows = [{"image_id": "a", "rle": "|1 1||"}, {"image_id": "b", "rle": "|||"}]
        image = np.zeros((256,1600,3),np.uint8)
        image[:,:,0] = np.arange(1600)[None]%256
        masks = np.zeros((4,256,1600),np.uint8); masks[1,40:60,700:710] = 1
        for epoch in [9,10]:
            reference = TrainDataset(rows,"patch",epoch=epoch,foreground_crops=True)
            augmented = AugmentedDataset(rows,epoch=epoch)
            with patch("phase2.data.native", side_effect=lambda row: (image, masks if row['image_id']=='a' else np.zeros_like(masks))):
                for index in range(2):
                    x,y = reference[index]; z,w = augmented[index]
                    self.assertTrue(torch.equal(y,w))
                    self.assertTrue(set(w.unique().tolist()) <= {0.,1.})
                    if index == 1: self.assertEqual(int(w.count_nonzero()), 0)
                    apply,b,c = augmented.augmentation_parameters[index]
                    self.assertTrue(torch.equal(z,photometric(x,b,c) if apply else x))

    def test_real_training_crops_and_negative_exposure_unchanged(self):
        rows = records("train"); base = TrainDataset(rows,"patch",epoch=9,foreground_crops=True)
        augmented = AugmentedDataset(rows,epoch=9)
        indices = [next(i for i,r in enumerate(rows) if r['rle'].split('|')[c]) for c in [0,1,3]]
        indices.append(next(i for i,r in enumerate(rows) if not any(r['rle'].split('|'))))
        self.assertEqual(len(augmented),9048)
        self.assertEqual(len({r['image_id'] for r in rows}),9048)
        self.assertEqual(sum(not any(r['rle'].split('|')) for r in rows),4250)
        for index in indices:
            _, truth = base[index]; image, changed = augmented[index]
            self.assertTrue(torch.equal(truth,changed))
            self.assertTrue(set(changed.unique().tolist()) <= {0.,1.})
            self.assertEqual(image.shape,(3,256,256))
            if not any(rows[index]['rle'].split('|')): self.assertEqual(int(changed.count_nonzero()),0)


if __name__ == "__main__": unittest.main(verbosity=2)
