# Post-evaluation qualitative error review

Reviewed after model/threshold locking and completion of held-out evaluation. No retraining, threshold tuning, label editing or sample exclusion followed this review.

Panels show input / supplied GT / prediction from top to bottom. Red, green, blue and yellow denote classes 1, 2, 3 and 4. Individual class scores do not include errors in other output channels.

| Case | Observed behavior | Class Dice | Panel |
|---|---|---:|---|
| class_1_worst (1e2bd7ad0.jpg) | Red annotated mark is missed; blue responses occur on nearby vertical structure. | 0.0000 | [Panel](../K_extended_patch/test/class_1_worst_1e2bd7ad0.png) |
| class_1_best (69205724f.jpg) | Compact annotated mark is localized; an extra blue class-3 halo remains. | 0.8476 | [Panel](../K_extended_patch/test/class_1_best_69205724f.png) |
| class_2_worst (268446e17.jpg) | Green annotation is missed; blue predictions follow a nearby steel-strip boundary. | 0.0000 | [Panel](../K_extended_patch/test/class_2_worst_268446e17.png) |
| class_2_best (2959d7fb1.jpg) | Green vertical annotation is substantially recovered, with blue spillover and isolated blue false positives elsewhere. | 0.6475 | [Panel](../K_extended_patch/test/class_2_best_2959d7fb1.png) |
| class_3_best (5f5e1e310.jpg) | Both long annotated vertical bands are closely localized; a spurious green response remains inside one band. | 0.9524 | [Panel](../K_extended_patch/test/class_3_best_5f5e1e310.png) |
| class_4_worst (aeda5b216.jpg) | The bright annotated yellow region is mainly predicted as blue: localization without the correct class is still an error. | 0.0013 | [Panel](../K_extended_patch/test/class_4_worst_aeda5b216.png) |
| class_4_best (d43c06a93.jpg) | The yellow annotated region at the left edge is largely recovered, but many blue predictions occur elsewhere. | 0.8471 | [Panel](../K_extended_patch/test/class_4_best_d43c06a93.png) |
| normal_false_alarm_1 (7f9df7221.jpg) | Supplied GT is empty. Strong fold-like texture and bright edges receive extensive blue/yellow predictions. This is a false alarm relative to the provided annotations; visual appearance alone cannot establish incorrect labels. | 0.0000 | [Panel](../K_extended_patch/test/normal_false_alarm_1_7f9df7221.png) |

Any-pixel false alarms occurred on 452 / 472 GT-empty images. This strict criterion includes tiny speckles; the largest illustrated case also contains extensive false-positive regions.
The selected model has class-2 precision 0.0556 and recall 0.1618. Four nonzero class scores do not establish a reliable four-class industrial detector.
The small class-2 test bucket contains only one GT component and none was detected. Its recall of zero must not be generalized to a large sample. The class-2 test partition contains only 20 positive images overall.
These observations describe the saved cases, not established causal explanations. Geometry tests and correctly aligned successful cases support coordinate correctness; they do not prove that every annotation is complete.
No instance matching, production acceptance threshold, multi-seed robustness or field validation is claimed.
