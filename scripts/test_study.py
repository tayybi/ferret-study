"""Focused checks for synthetic labels, leakage, and the adaptation gradient path."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from study import ChangeModel, FAST_NAMES, adapt, make_episode, objective, support


class StudyChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        y,x=np.mgrid[:32,:32]
        image=np.exp(-((x-16)**2+(y-15)**2)/150).astype(np.float32)
        self.volume=np.stack([image]*6)

    def test_generated_labels_and_repeatability(self):
        a=make_episode(self.volume,np.random.default_rng(3),4)
        b=make_episode(self.volume,np.random.default_rng(3),4)
        for x,y in zip(a,b):
            self.assertTrue(np.array_equal(x,y))
            self.assertTrue(np.isfinite(x).all())
        base,query,target,region=a
        self.assertEqual(base.shape,(4,1,32,32))
        self.assertTrue((target<=region).all())
        self.assertGreater(target[0].sum(),0)
        self.assertEqual(float(target[1].sum()),0)
        self.assertTrue((query>=0).all() and (query<=1).all())

    def test_adaptation_preserves_original_and_changes_only_adapters(self):
        model=ChangeModel()
        initial=copy.deepcopy(model.state_dict())
        fast=adapt(model,self.volume,np.random.default_rng(2),4,'cpu',2,.1)
        changed=[]
        for key,value in model.state_dict().items():
            self.assertTrue(torch.equal(value,initial[key]))
            if not torch.equal(fast.state_dict()[key],initial[key]):changed.append(key)
        self.assertTrue(changed)
        self.assertTrue(set(changed)<=set(FAST_NAMES))

    def test_first_order_outer_step_updates_shared_model(self):
        model=ChangeModel();initial=copy.deepcopy(model.state_dict())
        fast=adapt(model,self.volume,np.random.default_rng(1),4,'cpu')
        loss=objective(fast,make_episode(self.volume,np.random.default_rng(4),4),'cpu',
                       support(self.volume,np.random.default_rng(5),4,'cpu'))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        for original,temporary in zip(model.parameters(),fast.parameters()):
            if temporary.grad is not None:original.grad=temporary.grad.detach().clone()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.encoder.parameters()))
        optimizer.step()
        self.assertTrue(any(not torch.equal(v,initial[k]) for k,v in model.state_dict().items()))

    def test_manifest_splits_and_sources(self):
        study=Path(__file__).resolve().parents[1]
        manifest=json.loads((study/'data/manifest.json').read_text())
        rows=manifest['animals'];ids=[r['animal_id'] for r in rows]
        self.assertEqual(len(ids),len(set(ids)))
        sets={s:{r['animal_id'] for r in rows if r['split']==s} for s in ['train','validation','test']}
        for a,b in [('train','validation'),('train','test'),('validation','test')]:self.assertFalse(sets[a]&sets[b])
        self.assertTrue(all(r['label'].endswith('_baseline') for r in rows))
        self.assertTrue(all(r['protocol']=='04_RARE_T2_axial' for r in rows))
        with np.load(study/'data/baseline_t2.npz') as arrays:
            self.assertEqual(set(ids),set(arrays.files))
            self.assertTrue(all(np.isfinite(arrays[k]).all() for k in arrays.files))


if __name__=='__main__':unittest.main()
