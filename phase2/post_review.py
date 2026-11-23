"""Post-lock reporting and interleaved timing review; never trains or selects a model."""
import json
import shutil
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from .common import OUT, ROOT, IMAGES, load_model, seed_all, sha, write_json
from .data import records
from .patches import predict


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    status = read(OUT/"STATUS.json")
    if not status["phase2_complete"] or not read(OUT/"evidence_verification.json")["passed"]:
        raise ValueError("Finish the locked test and evidence checks before post-review")
    output = OUT/"post_review"
    if output.exists():
        raise FileExistsError("Preserve the existing post-review evidence")
    output.mkdir()
    lock = read(OUT/"final/lock.json")
    paths = [OUT/"final/model.pt",OUT/"final/lock.json",OUT/"partition.csv"]
    paths += list(OUT.glob("*/test/metrics.json"))+list((OUT/"ablations").glob("*/metrics.json"))
    frozen = {str(path.relative_to(OUT)):sha(path) for path in paths}
    shutil.copy2(OUT/"RESEARCH_REPORT.md",output/"RESEARCH_REPORT_before_review.md")
    seed_all(42)
    models, configs = {},[]
    for label in lock["candidates"]:
        model,checkpoint = load_model(OUT/label/"best_model.pt")
        models[label] = model
        thresholds = read(OUT/label/"thresholds.json")["thresholds"]
        configs.append((label,label,checkpoint["config"]["mode"],128,"hann",thresholds))
    threshold_c = read(OUT/"C_patch/thresholds.json")["thresholds"]
    configs += [("C_uniform_stride128","C_patch","patch",128,"uniform",threshold_c),
                ("C_hann_stride256","C_patch","patch",256,"hann",threshold_c)]
    rng = np.random.default_rng(42)
    rows = records("val")
    chosen = rng.choice(len(rows),20,replace=False)
    images = []
    for index in chosen:
        with Image.open(IMAGES/rows[index]["image_id"]) as f:
            images.append(np.asarray(f.convert("RGB")).copy())
    timings = {entry[0]:[] for entry in configs}

    def run(entry,image):
        label,model_label,mode,stride,kind,thresholds = entry
        probability = predict(models[model_label],image,mode,stride=stride,kind=kind)
        if not np.isfinite(probability).all():
            raise ValueError("Nonfinite timing prediction")
        return probability>=np.asarray(thresholds,dtype=np.float32)[:,None,None]

    for entry in configs:
        for _ in range(5):
            run(entry,images[0])
    for repeat in range(3):
        for image in images:
            for index in rng.permutation(len(configs)):
                entry = configs[index]
                torch.cuda.synchronize()
                started = time.perf_counter()
                result = run(entry,image)
                torch.cuda.synchronize()
                timings[entry[0]].append((time.perf_counter()-started)*1000)
                del result
        print(f"Interleaved timing repeat {repeat+1}/3 completed",flush=True)
    timing_result = {"seed":42,"validation_image_ids":[rows[index]["image_id"] for index in chosen],
                     "images":20,"repeats":3,"measurements_per_configuration":60,"warmups_per_configuration":5,
                     "configuration_order":"independently randomized for each image and repeat; same images for every configuration",
                     "scope":"decoded native RGB to binary masks, including transfers, reconstruction, finite-probability check and locked thresholds; excludes file IO, GT and metrics",
                     "all_eight_models_resident_on_GPU":True,"device_settings_changed":False,
                     "gpu":torch.cuda.get_device_name(),"torch_version":str(torch.__version__),
                     "note":"Supplementary latency measurement only. Memory figures remain the single-model test measurements. No model/threshold/selection is changed.",
                     "latency_ms":{name:{"mean":float(np.mean(v)),"median":float(np.median(v)),"p95":float(np.percentile(v,95)),"samples":v} for name,v in timings.items()}}
    write_json(output/"interleaved_runtime.json",timing_result)
    # Readability correction: put legends below the axes instead of over high recall bars.
    fig,axes = plt.subplots(1,2,figsize=(12,6.2))
    x = np.arange(4)
    width = .8/len(lock["candidates"])
    for j,label in enumerate(lock["candidates"]):
        m = read(OUT/label/"test/metrics.json")
        position = x+(j-(len(lock["candidates"])-1)/2)*width
        axes[0].bar(position,[m["per_class"][str(c)]["dice"] for c in range(1,5)],width,label=label)
        recalls = [m["performance_by_size"]["small"][str(c)]["object_recall"] for c in range(1,5)]
        axes[1].bar(position,[v if v is not None else np.nan for v in recalls],width,label=label)
    for ax,title in zip(axes,["Held-out per-class Dice","Small-component recall (>=10% GT pixels)\nAbsent GT size buckets have no bar"]):
        ax.set(xticks=x,xticklabels=["Class 1","Class 2","Class 3","Class 4"],ylim=(0,1),title=title)
        ax.legend(fontsize=8,ncol=2,loc="upper center",bbox_to_anchor=(.5,-.14),frameon=False)
    fig.subplots_adjust(bottom=.31,top=.88,left=.06,right=.98,wspace=.16)
    fig.savefig(output/"heldout_per_class_reviewed.png",dpi=180)
    plt.close(fig)
    examples = read(OUT/lock["selected_candidate"]/"test/examples.json")
    review_lines = ["# Post-evaluation qualitative error review","",
                    "Reviewed after model/threshold locking and completion of held-out evaluation. No retraining, threshold tuning, label editing or sample exclusion followed this review.","",
                    "Panels show input / supplied GT / prediction from top to bottom. Red, green, blue and yellow denote classes 1, 2, 3 and 4. Individual class scores do not include errors in other output channels.","",
                    "| Case | Observed behavior | Class Dice | Panel |","|---|---|---:|---|"]
    observations = {
        "class_1_best":"Compact annotated mark is localized; an extra blue class-3 halo remains.",
        "class_1_worst":"Red annotated mark is missed; blue responses occur on nearby vertical structure.",
        "class_2_best":"Green vertical annotation is substantially recovered, with blue spillover and isolated blue false positives elsewhere.",
        "class_2_worst":"Green annotation is missed; blue predictions follow a nearby steel-strip boundary.",
        "class_3_best":"Both long annotated vertical bands are closely localized; a spurious green response remains inside one band.",
        "class_4_best":"The yellow annotated region at the left edge is largely recovered, but many blue predictions occur elsewhere.",
        "class_4_worst":"The bright annotated yellow region is mainly predicted as blue: localization without the correct class is still an error.",
        "normal_false_alarm_1":"Supplied GT is empty. Strong fold-like texture and bright edges receive extensive blue/yellow predictions. This is a false alarm relative to the provided annotations; visual appearance alone cannot establish incorrect labels."}
    reviewed_ids = {"class_1_best":"69205724f.jpg","class_1_worst":"1e2bd7ad0.jpg",
                    "class_2_best":"2959d7fb1.jpg","class_2_worst":"268446e17.jpg",
                    "class_3_best":"5f5e1e310.jpg","class_4_best":"d43c06a93.jpg",
                    "class_4_worst":"aeda5b216.jpg","normal_false_alarm_1":"7f9df7221.jpg"}
    reviewed_checkpoint = "0ce6c174d27fff24551faa7ce94f7a2ec7f9a98294a0eac7746a8c61b89eed59"
    if lock["checkpoint_sha256"] != reviewed_checkpoint:
        review_lines += ["","The original study's qualitative notes are tied to its reviewed checkpoint and image IDs. This reproduction has a different checkpoint hash; inspect its panels before adding new visual observations. The quantitative results below are computed from this reproduction's own saved metrics.",""]
    for example in examples:
        if (lock["checkpoint_sha256"] == reviewed_checkpoint and
            example["image_id"] == reviewed_ids.get(example["label"])):
            file = example["file"]
            review_lines.append(f"| {example['label']} ({example['image_id']}) | {observations[example['label']]} | {example['example_dice']:.4f} | [Panel](../{lock['selected_candidate']}/test/{file}) |")
    errors = read(OUT/"image_level_errors.json")[lock["selected_candidate"]]
    selected_metrics = read(OUT/lock["selected_candidate"]/"test/metrics.json")
    class_two = selected_metrics["per_class"]["2"]
    small_two = selected_metrics["performance_by_size"]["small"]["2"]
    review_lines += ["",f"Any-pixel false alarms occurred on {errors['normal_false_alarm_images']} / {errors['normal_image_count']} GT-empty images. This strict criterion includes tiny speckles; the largest illustrated case also contains extensive false-positive regions.",
                     f"The selected model has class-2 precision {class_two['precision']} and recall {class_two['recall']}. Four nonzero class scores do not establish a reliable four-class industrial detector.",
                     f"The small class-2 test bucket contains {small_two['objects']} GT components, of which {small_two['detected_objects']} were detected. This small sample does not support a broad small-defect generalization. The class-2 test partition contains {errors['gt_positive_images_by_class'][1]} positive images overall.",
                     "These observations describe the saved cases, not established causal explanations. Geometry tests and correctly aligned successful cases support coordinate correctness; they do not prove that every annotation is complete.",
                     "No instance matching, production acceptance threshold, multi-seed robustness or field validation is claimed."]
    (output/"QUALITATIVE_ERROR_ANALYSIS.md").write_text("\n".join(review_lines)+"\n",encoding="utf-8")
    assert all(sha(OUT/path)==digest for path,digest in frozen.items())
    write_json(output/"review_provenance.json",{"model_selection_unchanged":True,"checkpoint_lock_partition_and_test_metrics_unchanged":True,
               "frozen_input_sha256":frozen,"post_review_source_sha256":sha(ROOT/"phase2/post_review.py"),
               "review_inputs":"fixed held-out output panels and metrics; timing-only inputs from validation; no model selection or training"})
    shutil.copy2(ROOT/"phase2/post_review.py",output/"post_review.py")
    report = (OUT/"RESEARCH_REPORT.md").read_text(encoding="utf-8")
    report = report.replace("The measured patch pipeline does not improve macro Dice over B on these fixed held-out samples.",
                            "The matched reference C patch pipeline does not improve macro Dice over B on these fixed held-out samples.")
    appendix = ["","## Post-evaluation review and timing control","",
                "The selected K pipeline exceeds B on the common held-out split, but K also changes loss, training crops and budget. The negative matched C/B result above concerns patching alone.",
                "Sequential test timings varied even between identical dense-network configurations. Do not attribute those timing differences to learned weights. The supplementary benchmark preloads 20 validation images, interleaves randomized configuration order for three repeats, and excludes GT/metric work. All eight models stay resident; single-model memory figures remain in the original test results.","",
                "| Configuration | Interleaved mean ms | Median ms | P95 ms |","|---|---:|---:|---:|"]
    for label,value in timing_result["latency_ms"].items():
        appendix.append(f"| {label} | {value['mean']:.2f} | {value['median']:.2f} | {value['p95']:.2f} |")
    appendix += ["","[Detailed timing protocol and raw samples](post_review/interleaved_runtime.json). No device settings or model parameters changed.",
                 "[Reviewed error cases and limitations](post_review/QUALITATIVE_ERROR_ANALYSIS.md). This is post-evaluation description, not a new model-selection step.",
                 "![Per-class results with unobstructed legends](post_review/heldout_per_class_reviewed.png)"]
    (OUT/"RESEARCH_REPORT.md").write_text(report+"\n".join(appendix)+"\n",encoding="utf-8")
    status["post_evaluation_review_completed"] = True
    status["interleaved_runtime_benchmark_completed"] = True
    status["normal_false_alarm_images"] = errors["normal_false_alarm_images"]
    status["normal_images"] = errors["normal_image_count"]
    write_json(OUT/"STATUS.json",status)
    print("Post-evaluation review and interleaved timing completed; locked evidence unchanged.",flush=True)


if __name__ == "__main__":
    main()
