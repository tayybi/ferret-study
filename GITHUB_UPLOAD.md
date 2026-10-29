# Repository scope

This repository contains the `ferret-study` analysis project: scripts, the
ISBI 2027 manuscript and figures, analysis plans, reports, saved results,
prepared data, checkpoints, and reference resources with their existing
provenance documentation.

The separate `Ferret/` scanner dataset (approximately 92 GiB) is not included.
Scripts that read original acquisitions expect that folder next to this
repository:

```text
workspace/
  Ferret/
  ferret-study/
```

The local Python environment, Python caches, logs, credentials in `.env`
files, and LaTeX build intermediates are ignored. Manuscript PDFs and
Overleaf archives are retained.

The intended GitHub repository is private and named `ferret-study`.
The largest included file is
`blast-pipeline/models/main/synthetic.npz` (approximately 91 MiB).
This initial research snapshot includes generated artifacts, so cloning
requires substantially more storage than a source-only repository.

See `README.md` for the workflow and study limitations, `environment.json`
and `full-pipeline/project-packages.txt` for environment information, and
`reference/README.md` for reference-atlas provenance and attribution.
