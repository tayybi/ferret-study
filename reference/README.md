# Public adult-ferret reference

Downloaded on 26 October 2026 from the [Scalable Brain Atlas adult ferret page](https://scalablebrainatlas.incf.org/ferret/HSRetal17).

Original atlas reference: Hutchinson EB et al. (2017), *Population based MRI and DTI templates of the adult ferret brain and tools for voxelwise analysis*, NeuroImage 152:575–589. https://doi.org/10.1016/j.neuroimage.2017.03.009

Files:

- `Template4D_TE040.nii.gz`: multi-echo structural template, TE40.
- `evDTI_SEGMENTATION.nii.gz`: anatomical labels in the atlas coordinate grid.

Downloaded originals are preserved. The pipeline uses the filled union of nonzero labels as a provisional mask, reorders the reference axes to the inspected scanner image convention, and transfers it using rigid registration. This combines in-vivo structural contrast with label boundaries originally defined using ex-vivo data; matching grids do not establish expert-validated alignment. Mask boundary omissions and cross-template differences remain limitations. No individual anatomical region labels are analyzed.

The initial axis convention and approximate centering were inspected on training animal BCF100. The retained rigid transfer was reviewed on contact sheets for all baseline animals. These visual checks are engineering QA, not expert anatomical validation. The exploratory registration trial files record this development step and are not study outcomes.

The source page identifies atlas-specific terms and a citation policy. Retain attribution to the atlas authors and Scalable Brain Atlas when using these resources. No atlas data have been published or redistributed outside this local project.
