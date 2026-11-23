# Licensing and attribution review

Reviewed 2026-11-12. This records open questions; it does not grant rights or constitute a legal determination.

## Project source

No LICENSE file or license identifier has been applied. MIT is recommended as a simple permissive option **only after** the owner confirms rights to the included source and approves that license. Until then, do not describe this repository as MIT-licensed or open-source-licensed. Publishing source without a license does not itself grant general reuse rights.

The preserved classification project contains AI-assistance disclosures and unresolved ownership/source questions. Those classification files are excluded, but they do not establish verified authorship of every segmentation file. The segmentation source has no discovered file-level external attribution/license headers. The owner must confirm whether code was adapted from tutorials, repositories, coursework templates, or collaborators and add required attribution before public publication.

This repository preparation used AI assistance for auditing, code adaptation, new inference/tests, documentation and verification. It does not claim that all original source was independently authored or that missing permissions were resolved. Source paths/hashes and changes are documented in source_provenance.json and experiments.md.

## Dataset and derived artifacts

Official source: [Kaggle Severstal Steel Defect Detection](https://www.kaggle.com/competitions/severstal-steel-defect-detection/data). The [competition rules](https://www.kaggle.com/competitions/severstal-steel-defect-detection/rules) govern access and use; the rules page's indexed text limits data use to competition participation and Kaggle forums. The dynamic page was not fully extractable during this review. Do not assume general research, commercial, public-image or checkpoint redistribution rights. Verify the complete applicable rules in the authenticated Kaggle interface and obtain clarification from the rights holder as needed.

Images, train.csv, RLE manifests, raw prediction panels and trained checkpoints are withheld. Aggregate numerical metrics, configurations and plots of those metrics are proposed for publication; the owner should confirm that these research outputs and the underlying use are permitted before public release. A code license cannot override dataset restrictions.

## Direct dependencies

Dependencies are installed separately; their source or wheel binaries are not redistributed here. Record and honor their notices if bundling them in a future application/container. Package wheels may carry additional notices for bundled components.

| Dependency | Upstream license source | Review |
|---|---|---|
| PyTorch | [LICENSE](https://github.com/pytorch/pytorch/blob/main/LICENSE) | BSD-style project terms plus component notices; inspect the installed distribution for exact terms |
| NumPy | [LICENSE.txt](https://github.com/numpy/numpy/blob/main/LICENSE.txt) | BSD-3-Clause project terms; binary dependencies have additional notices |
| Pillow | [LICENSE](https://github.com/python-pillow/Pillow/blob/main/LICENSE) | MIT-CMU / historical PIL-related terms; retain applicable notices if redistributed |
| Matplotlib (optional plotting only) | [LICENSE](https://github.com/matplotlib/matplotlib/blob/main/LICENSE/LICENSE) | Matplotlib license and component notices; installed separately |

The U-Net paper is cited as an architectural reference: [Ronneberger, Fischer and Brox (2015)](https://arxiv.org/abs/1505.04597). No external pretrained weights are used in the included segmentation architecture. The old optional ResNet18 classifier is not included.

## Decisions still needed

1. Confirm ownership and external source attribution for included segmentation code.
2. Approve MIT or choose another software license after that confirmation.
3. Confirm permitted dataset use and publication of aggregate research outputs.
4. Resolve image/checkpoint redistribution rights separately before adding them.
