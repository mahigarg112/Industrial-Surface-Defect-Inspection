# Selected model card

K_extended_patch, epoch9; SmallUNet,482788 parameters; 3 RGB inputs,4 independent sigmoid channels. Original model SHA256: 0ce6c174d27fff24551faa7ce94f7a2ec7f9a98294a0eac7746a8c61b89eed59. Exact locked thresholds: [0.949999988079071, 0.9750000238418579, 0.5, 0.949999988079071].

Native256x256 tiles, stride128, positive Hann probability blending, at most8 patches per CUDA batch; original1600x256 gives12 tiles, final x1344. Masks remain aligned to the decoded image. Uploaded geometries beyond the study size are supported mechanically but not accuracy-validated. RGB/255; no ImageNet normalization or external pretrained encoder.

Selected on validation only. Tuned validation Dice0.239229/IoU0.141876; independent held-out1006 images Dice0.233996/IoU0.137937. Training split9048, original validation2514, new test1006; original Phase I weights contaminate that new test and are not a valid held-out baseline.

Class | Dice | IoU | Precision | Recall
1 | 0.190200 | 0.105095 | 0.188940 | 0.191477
2 | 0.082752 | 0.043162 | 0.055592 | 0.161799
3 | 0.415618 | 0.262322 | 0.397702 | 0.435224
4 | 0.247413 | 0.141170 | 0.206813 | 0.307846

Class2 remains weak. Any-pixel false alarms452/472 GT-empty images (strict criterion, includes speckles; some errors are extensive). Small class2 test bucket contains only1 component, none recovered at the declared>=10% criterion. These results do not establish industrial reliability. Do not use for acceptance/safety decisions.

K uses different loss/crops and9 rather than3 epochs. Its improvement cannot be attributed to patching alone. Matched patch-only C underperformed whole-image B. Only one training seed; no second full training replication; near-duplicate/coil groups and external-domain robustness unverified. Validation selection/tuning is optimistic; held-out test has already been opened and cannot drive new settings.

CPU fallback uses the same weights and full native coordinates; FP32 CPU versus AMP CUDA can change a few boundary pixels. No new accuracy estimate is made for release smoke images or user-adjusted thresholds. Areas are per-class mask fractions, not confidence. Channels can overlap.

Research inference timing K mean70.33ms (sequential) and65.54ms (randomized interleaved review,median58.75ms,p9598.83ms). These exclude cold model loading and disk IO; app processing timing includes decode/inference/current rendering. Inference allocated/reserved119413248/153092096 bytes; training404252672/476053504 bytes, excluding other WDDM usage. See raw protocols rather than interpreting cold app timing as a benchmark.

Sources: phase2/artifacts/final/lock.json, K_extended_patch/test/metrics.json, image_level_errors.json, post_review/interleaved_runtime.json, RESEARCH_REPORT.md; release/evidence records.
