import os
os.environ.setdefault('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS','2')
os.environ.setdefault('MPLCONFIGDIR','/tmp/ferret-mpl')
import unittest
import numpy as np
import ants
from scipy.ndimage import binary_erosion
from full_pipeline import scan_image, ROOT, FULL
from study import make_episode
import json

class FullChecks(unittest.TestCase):
 def test_transform_direction_and_nearest_mask(self):
  a=np.zeros((24,24,12),dtype=np.float32);a[10:14,10:14,4:8]=1
  moving=ants.from_numpy(a)
  fixed=ants.from_numpy(np.zeros_like(a))
  transform=ants.create_ants_transform('AffineTransform',dimension=3,translation=(2,0,0))
  path='/tmp/ferret-known-transform.mat';ants.write_transform(transform,path)
  warped=ants.apply_transforms(fixed,moving,[path],interpolator='nearestNeighbor').numpy()
  self.assertEqual(np.argmax(warped.sum((1,2))),8)
  self.assertEqual(set(np.unique(warped)),{0.,1.})
 def test_source_geometry_spacing(self):
  rows=json.loads((FULL/'data/manifest.json').read_text())['animals'];row=rows[0]
  im,p=scan_image(ROOT/row['source'])
  self.assertTrue(np.allclose(im.spacing[:2],[(55/366)*365/159]*2))
  self.assertAlmostEqual(im.spacing[2],1.5)
  positions=np.array([float(x) for x in p['VisuCorePosition'].split()]).reshape(-1,3)
  self.assertTrue(np.allclose(im.origin,positions[0]))
 def test_synthetic_changes_stay_in_mask(self):
  with np.load(FULL/'data/baseline_t2.npz') as f:volumes={k:f[k] for k in f.files}
  with np.load(FULL/'data/brain_masks.npz') as f:masks={k:f[k] for k in f.files}
  for animal in volumes:
   arrays=make_episode(volumes[animal],np.random.default_rng(5),4,region_masks=masks[animal])
   base,query,target,region=arrays
   self.assertTrue((target<=region).all())
   self.assertTrue((region.sum((1,2,3))>30).all())
   self.assertEqual(float(target[1].sum()),0)
if __name__=='__main__':unittest.main()
