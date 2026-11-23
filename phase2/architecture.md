# Architecture and tensor coordinates

The same unchanged Phase I SmallUNet is used for B and C. N is batch size, normally 8.
Four raw output logits use independent sigmoid probabilities; this is not a four-way softmax.

| Tensor | B whole-image shape | C native-patch shape |
|---|---|---|
| Input RGB | N×3×128×512 | N×3×256×256 |
| enc1: two 3×3 conv+BN+ReLU | N×16×128×512 | N×16×256×256 |
| pool → enc2 | N×32×64×256 | N×32×128×128 |
| pool → enc3 | N×64×32×128 | N×64×64×64 |
| pool → bottleneck | N×128×16×64 | N×128×32×32 |
| up3 + concat enc3 → dec3 | N×64×32×128 | N×64×64×64 |
| up2 + concat enc2 → dec2 | N×32×64×256 | N×32×128×128 |
| up1 + concat enc1 → dec1 | N×16×128×512 | N×16×256×256 |
| 1×1 output conv | N×4×128×512 | N×4×256×256 |
| Sigmoid + native mapping | 4×256×1600 | 4×256×1600 after blending |

```mermaid
flowchart LR
  I[Native RGB 3 x 256 x 1600] --> T[12 overlapping native patches]
  T --> E1[enc1: 16 channels]
  E1 --> E2[pool + enc2: 32]
  E2 --> E3[pool + enc3: 64]
  E3 --> B[pool + bottleneck: 128]
  B --> D3[up + concat + dec3: 64]
  E3 --> D3
  D3 --> D2[up + concat + dec2: 32]
  E2 --> D2
  D2 --> D1[up + concat + dec1: 16]
  E1 --> D1
  D1 --> O[Four patch sigmoid probabilities]
  O --> R[Positive Hann blend in native coordinates]
  R --> P[4 x 256 x 1600 probabilities]
  P --> M[Four validation-locked thresholds]
```

Patch coordinates are half-open `(y,x,bottom,right)` in the native image. The pixel `(dy,dx)`
within a patch maps to `(y+dy,x+dx)`. Probabilities are weighted by a separable positive Hann window,
accumulated at those coordinates, and divided by the accumulated weights. There is no coordinate
rounding or resize in the native branch. A final edge-aligned patch covers columns 1344–1599.
The non-overlap ablation still edge-aligns its last patch, so the last 192 columns overlap: complete
coverage takes priority, and it is labeled *stride-256 minimal overlap*, not strictly zero overlap.
