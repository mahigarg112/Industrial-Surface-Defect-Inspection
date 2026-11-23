"""Synthetic analytical/numerical checks, executed before any training."""
from pathlib import Path
import io
import unittest
import numpy as np
import torch
from phase2.common import write_json
from phase2.objectives import balanced_present_dice_loss
from .loss import pairwise_probability_overlap,loss_terms


class OverlapTests(unittest.TestCase):
    def test_disjoint_zero(self):
        p=torch.zeros(2,4,5,6,requires_grad=True)
        with torch.no_grad():
            p[0,0,:,0]=1;p[0,1,:,1]=.8;p[1,3,:,:]=1
        loss=pairwise_probability_overlap(p)
        self.assertEqual(loss.item(),0);loss.backward();self.assertTrue(torch.isfinite(p.grad).all())

    def test_known_pairs_and_symmetry(self):
        p=torch.tensor([.9,.9,0.,0.]).reshape(1,4,1,1)
        self.assertAlmostEqual(pairwise_probability_overlap(p).item(),.81/6,places=6)
        self.assertAlmostEqual(pairwise_probability_overlap(torch.ones_like(p)).item(),1,places=6)
        self.assertEqual(pairwise_probability_overlap(p).item(),pairwise_probability_overlap(p[:,[3,1,0,2]]).item())

    def test_numerical_gradient(self):
        generator=torch.Generator().manual_seed(42)
        p=torch.rand((1,4,2,2),generator=generator,dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(pairwise_probability_overlap,(p,)))

    def test_logit_chain_rule_and_extremes(self):
        z=torch.tensor([-.7,.2,1.,-2.]).reshape(1,4,1,1).requires_grad_()
        p=z.sigmoid();loss=pairwise_probability_overlap(p);loss.backward()
        expected=p*(1-p)*(p.sum(1,keepdim=True)-p)/6
        self.assertTrue(torch.allclose(z.grad,expected,atol=1e-7))
        logits=torch.full((1,4,2,2),-1000.,requires_grad=True)
        with torch.no_grad():logits[:,0]=1000
        penalty=pairwise_probability_overlap(logits.sigmoid())
        self.assertEqual(penalty.item(),0);penalty.backward();self.assertTrue(torch.isfinite(logits.grad).all())

    def test_zero_coefficient_and_targets(self):
        z=torch.linspace(-3,3,64).reshape(1,4,4,4).requires_grad_()
        target=torch.zeros_like(z);target[:,1,1:3,1:3]=1;before=target.clone();weights=torch.tensor([14.,51.,2.6,15.7])
        base=balanced_present_dice_loss(z,target,weights)
        zero,_,_=loss_terms(z,target,weights,0.)
        self.assertTrue(torch.equal(base,zero));total,b,aux=loss_terms(z,target,weights,.1)
        self.assertTrue(torch.allclose(total,b+.1*aux));total.backward()
        self.assertTrue(torch.isfinite(z.grad).all());self.assertTrue(torch.equal(target,before))
        for coefficient in [-1.,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):loss_terms(z,target,weights,coefficient)

    def test_cuda_amp_and_shape(self):
        self.assertTrue(torch.cuda.is_available())
        z=torch.linspace(-50,50,256,device='cuda',dtype=torch.float16).reshape(2,4,4,8).requires_grad_()
        target=torch.zeros_like(z);target[:,2,:,2]=1
        with torch.autocast('cuda',dtype=torch.float16):total,base,aux=loss_terms(z,target,torch.ones(4,device='cuda'),.1)
        self.assertEqual(total.dtype,torch.float32);total.backward()
        self.assertTrue(torch.isfinite(total));self.assertTrue(torch.isfinite(aux));self.assertTrue(torch.isfinite(z.grad).all())
        with self.assertRaises(ValueError):pairwise_probability_overlap(torch.zeros(1,3,2,2))


if __name__=='__main__':
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(OverlapTests))
    here=Path(__file__).resolve().parent
    (here/'unit_tests.log').write_text(stream.getvalue(),encoding='utf-8');print(stream.getvalue())
    write_json(here/'unit_tests.json',{'passed':result.wasSuccessful(),'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped)})
    if not result.wasSuccessful():raise SystemExit(1)
