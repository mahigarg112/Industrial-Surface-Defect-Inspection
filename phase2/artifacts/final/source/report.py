"""Export measured research evidence; no performance claims without completed artifacts."""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .audit import verify
from .common import OUT, load_model, sha, write_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def paired_bootstrap(a,b,repeats=2000):
    if [r["image_id"] for r in a] != [r["image_id"] for r in b]:
        raise ValueError("Candidates did not evaluate the same ordered test samples")
    arrays = [np.array([r["class_counts_tp_fp_fn"] for r in rows],dtype=np.int64) for rows in [a,b]]
    def macro(total):
        tp,fp,fn = total
        return np.mean(2*tp/(2*tp+fp+fn))
    rng = np.random.default_rng(42)
    differences = []
    for _ in range(repeats):
        index = rng.integers(0,len(a),len(a))
        sums = [arr[index].sum(0) for arr in arrays]
        if any(np.any((s[0]+s[2])==0) for s in sums):
            continue
        differences.append(macro(sums[1])-macro(sums[0]))
    return {"delta_C_minus_B":float(macro(arrays[1].sum(0))-macro(arrays[0].sum(0))),
            "percentile_95_interval":[float(v) for v in np.percentile(differences,[2.5,97.5])],
            "bootstrap_repeats":repeats,"valid_repeats":len(differences),"seed":42,
            "scope":"Paired image bootstrap of fixed trained models, not training-seed uncertainty; no model choice uses this result."}


def generate():
    for filename in ["pipeline_tests.json","evaluation_tests.json","smoke_results.json"]:
        if not read(OUT/filename)["passed"]:
            raise ValueError(f"Required check did not pass: {filename}")
    architecture = read(OUT/"architecture_verified.json")
    original = read(OUT/"A_phase1_original_validation_cpu.json")
    lock = read(OUT/"final/lock.json")
    if "F_present_dice_patch" in lock["candidates"] and not read(OUT/"present_preflight.json")["passed"]:
        raise ValueError("Present-Dice GPU preflight failed")
    if "G_present_dice_whole" in lock["candidates"] and not read(OUT/"present_whole_preflight.json")["passed"]:
        raise ValueError("Whole present-Dice GPU preflight failed")
    if "H_foreground_patch" in lock["candidates"]:
        if not read(OUT/"foreground_preflight.json")["passed"] or not read(OUT/"foreground_sampling_tests.json")["passed"]:
            raise ValueError("Foreground sampler preflight failed")
    if "J_balanced_dice_patch" in lock["candidates"] and not read(OUT/"balanced_dice_preflight.json")["passed"]:
        raise ValueError("Balanced-Dice preflight failed")
    if "K_extended_patch" in lock["candidates"] and not read(OUT/"resume_preflight.json")["passed"]:
        raise ValueError("Continuation preflight failed")
    if sha(OUT/"final/model.pt") != lock["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed")
    model,checkpoint = load_model(OUT/"final/model.pt")
    del model
    names = {"B_whole":OUT/"B_whole/test","C_patch":OUT/"C_patch/test",
             "C_uniform_stride128":OUT/"ablations/C_uniform_stride128",
             "C_hann_stride256":OUT/"ablations/C_hann_stride256"}
    if "E_weighted_patch" in lock["candidates"]:
        names["E_weighted_patch"] = OUT/"E_weighted_patch/test"
    if "F_present_dice_patch" in lock["candidates"]:
        names["F_present_dice_patch"] = OUT/"F_present_dice_patch/test"
    if "G_present_dice_whole" in lock["candidates"]:
        names["G_present_dice_whole"] = OUT/"G_present_dice_whole/test"
    if "H_foreground_patch" in lock["candidates"]:
        names["H_foreground_patch"] = OUT/"H_foreground_patch/test"
    if "J_balanced_dice_patch" in lock["candidates"]:
        names["J_balanced_dice_patch"] = OUT/"J_balanced_dice_patch/test"
    if "K_extended_patch" in lock["candidates"]:
        names["K_extended_patch"] = OUT/"K_extended_patch/test"
    metrics = {name:read(directory/"metrics.json") for name,directory in names.items()}
    for value in metrics.values():
        if value["images"] != 1006 or len(value["per_class"]) != 4 or value["partition_sha256"] != lock["partition_sha256"]:
            raise ValueError("Incomplete/incompatible evaluation")
    for label in lock["candidates"]:
        examples = read(OUT/label/"test/examples.json")
        if {example["focus_class"] for example in examples if example["focus_class"] is not None} != {1,2,3,4}:
            raise ValueError("Representative examples do not cover all four classes")
    inference = read(OUT/"new_image_inference/inference.json")
    if inference["checkpoint_sha256"] != lock["checkpoint_sha256"]:
        raise ValueError("Inference did not use selected checkpoint")
    verify()
    bootstrap = paired_bootstrap(read(OUT/"B_whole/test/per_image_errors.json"),read(OUT/"C_patch/test/per_image_errors.json"))
    write_json(OUT/"paired_uncertainty.json",bootstrap)
    objective_bootstrap = None
    if "E_weighted_patch" in names:
        objective_bootstrap = paired_bootstrap(read(OUT/"C_patch/test/per_image_errors.json"),read(OUT/"E_weighted_patch/test/per_image_errors.json"))
        objective_bootstrap["delta_E_minus_C"] = objective_bootstrap.pop("delta_C_minus_B")
        write_json(OUT/"paired_uncertainty_objective.json",objective_bootstrap)
    present_bootstrap = None
    if "F_present_dice_patch" in names:
        present_bootstrap = paired_bootstrap(read(OUT/"C_patch/test/per_image_errors.json"),read(OUT/"F_present_dice_patch/test/per_image_errors.json"))
        present_bootstrap["delta_F_minus_C"] = present_bootstrap.pop("delta_C_minus_B")
        write_json(OUT/"paired_uncertainty_present_dice.json",present_bootstrap)
    representation_present = None
    if "F_present_dice_patch" in names and "G_present_dice_whole" in names:
        representation_present = paired_bootstrap(read(OUT/"G_present_dice_whole/test/per_image_errors.json"),read(OUT/"F_present_dice_patch/test/per_image_errors.json"))
        representation_present["delta_F_patch_minus_G_whole"] = representation_present.pop("delta_C_minus_B")
        write_json(OUT/"paired_uncertainty_representation_present.json",representation_present)
    sampling_bootstrap = None
    if "H_foreground_patch" in names:
        sampling_bootstrap = paired_bootstrap(read(OUT/"F_present_dice_patch/test/per_image_errors.json"),read(OUT/"H_foreground_patch/test/per_image_errors.json"))
        sampling_bootstrap["delta_H_minus_F"] = sampling_bootstrap.pop("delta_C_minus_B")
        write_json(OUT/"paired_uncertainty_foreground_sampling.json",sampling_bootstrap)
    balance_bootstrap = None
    if "J_balanced_dice_patch" in names:
        balance_bootstrap = paired_bootstrap(read(OUT/"H_foreground_patch/test/per_image_errors.json"),read(OUT/"J_balanced_dice_patch/test/per_image_errors.json"))
        balance_bootstrap["delta_J_minus_H"] = balance_bootstrap.pop("delta_C_minus_B")
        write_json(OUT/"paired_uncertainty_class_balance.json",balance_bootstrap)
    budget_bootstrap = None
    if "K_extended_patch" in names:
        extended_config = read(OUT/"K_extended_patch/config.json")
        parent_label = extended_config["continuation_parent"]
        budget_bootstrap = paired_bootstrap(read(OUT/parent_label/"test/per_image_errors.json"),read(OUT/"K_extended_patch/test/per_image_errors.json"))
        budget_bootstrap["delta_K_minus_parent"] = budget_bootstrap.pop("delta_C_minus_B")
        budget_bootstrap["parent"] = parent_label
        write_json(OUT/"paired_uncertainty_training_budget.json",budget_bootstrap)
    image_errors = {}
    for name,directory in names.items():
        errors = read(directory/"per_image_errors.json")
        values = np.asarray([r["class_counts_tp_fp_fn"] for r in errors],dtype=np.int64)
        positive = (values[:,0,:]+values[:,2,:])>0
        predicted = (values[:,0,:]+values[:,1,:])>0
        normal = ~positive.any(1)
        alarms = int((normal & predicted.any(1)).sum())
        image_errors[name] = {"normal_image_count":int(normal.sum()),"normal_false_alarm_images":alarms,
                             "normal_false_alarm_rate_any_predicted_pixel":alarms/int(normal.sum()) if normal.any() else None,
                             "gt_positive_images_by_class":positive.sum(0).tolist(),
                             "complete_gt_miss_images_by_class":(positive & (values[:,0,:]==0)).sum(0).tolist(),
                             "definition":"normal false alarm = any predicted class pixel on a GT-empty image; complete class miss = zero recovered GT pixels on a class-positive image"}
    write_json(OUT/"image_level_errors.json",image_errors)
    rows = []
    for name,m in metrics.items():
        train_label = name if name in lock["candidates"] else "C_patch"
        train_info = read(OUT/train_label/"run_info.json")
        row = {"experiment":name,"macro_dice":m["macro_dice"],"macro_iou":m["macro_iou"],
               "mean_latency_ms":m["latency_ms"]["mean"],"p95_latency_ms":m["latency_ms"]["p95"],
               "peak_allocated_bytes":m["peak_allocated_bytes"],"parameters":m["parameters"],
               "peak_reserved_bytes":m["peak_reserved_bytes"],
               "training_peak_allocated_bytes":train_info["peak_allocated_bytes"],
               "training_peak_reserved_bytes":train_info["peak_reserved_bytes"],
               "training_epochs_completed":train_info["epochs_completed"],"selected_epoch":train_info["best_epoch"],
               "successful_training_updates":train_info["optimizer_steps"],
               "attempted_training_updates":train_info.get("attempted_optimizer_steps",train_info["optimizer_steps"]),
               "training_and_validation_duration_seconds":train_info["duration_seconds"]}
        row["continuation_only_seconds"] = train_info.get("continuation_only_seconds")
        row["normal_false_alarm_images"] = image_errors[name]["normal_false_alarm_images"]
        row["normal_false_alarm_rate"] = image_errors[name]["normal_false_alarm_rate_any_predicted_pixel"]
        for c in range(1,5):
            row.update({f"class_{c}_{key}":m["per_class"][str(c)][key] for key in ["dice","iou","precision","recall","false_positive_pixels","false_negative_pixels"]})
        rows.append(row)
    with (OUT/"comparison.csv").open("w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    exposure_rows = []
    for label in lock["candidates"]:
        for file in sorted((OUT/label).glob("epoch_*_target_exposure.json")):
            value = read(file)
            row = {"candidate":label,"epoch":int(file.stem.split("_")[1]),"training_samples":value["samples"]}
            for c in range(4):
                row[f"class_{c+1}_positive_samples"] = value["positive_samples_by_class"][c]
                row[f"class_{c+1}_positive_pixels"] = value["positive_pixels_by_class"][c]
            exposure_rows.append(row)
    with (OUT/"training_target_exposure.csv").open("w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f,fieldnames=list(exposure_rows[0]))
        writer.writeheader()
        writer.writerows(exposure_rows)
    fig,axes = plt.subplots(1,2,figsize=(11,4))
    for name in ["B_whole","C_patch"]:
        with (OUT/name/"history.csv").open(newline="") as f:
            history = list(csv.DictReader(f))
        epochs = [int(r["epoch"]) for r in history]
        axes[0].plot(epochs,[float(r["train_loss"]) for r in history],"o-",label=name)
        axes[1].plot(epochs,[float(r["val_macro_dice"]) for r in history],"o-",label=name)
    for ax,title in zip(axes,["Training BCE + soft Dice loss","Native validation macro Dice at 0.5"]):
        ax.set(xlabel="Epoch",title=title)
        ax.legend()
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT/"training_curves.png",dpi=180)
    plt.close(fig)
    for variant in [label for label in lock["candidates"] if label in ["E_weighted_patch","F_present_dice_patch","G_present_dice_whole","H_foreground_patch","J_balanced_dice_patch","K_extended_patch"]]:
        with (OUT/variant/"history.csv").open(newline="") as f:
            history = list(csv.DictReader(f))
        fig,axes = plt.subplots(1,2,figsize=(11,4))
        epochs = [int(r["epoch"]) for r in history]
        axes[0].plot(epochs,[float(r["train_loss"]) for r in history],"o-")
        axes[1].plot(epochs,[float(r["val_macro_dice"]) for r in history],"o-")
        axes[0].set(title=f"{variant} objective loss (scale differs from B/C)",xlabel="Epoch")
        axes[1].set(title=f"{variant} native validation Dice at 0.5",xlabel="Epoch")
        fig.tight_layout()
        fig.savefig(OUT/f"{variant}_training_curves.png",dpi=180)
        plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(11,4))
    x = np.arange(4)
    width = .8/len(lock["candidates"])
    for j,name in enumerate(lock["candidates"]):
        m = metrics[name]
        offset = (j-(len(lock["candidates"])-1)/2)*width
        axes[0].bar(x+offset,[m["per_class"][str(c)]["dice"] for c in range(1,5)],width=width,label=name)
        recalls = [m["performance_by_size"]["small"][str(c)]["object_recall"] for c in range(1,5)]
        axes[1].bar(x+offset,[value if value is not None else np.nan for value in recalls],width=width,label=name)
    for ax,title in zip(axes,["Held-out per-class Dice","Small-component recall (>=10% pixels recovered)\nNo bar when the GT size bucket is absent"]):
        ax.set(xticks=x,xticklabels=["Class 1","Class 2","Class 3","Class 4"],ylim=(0,1),title=title)
        ax.legend(fontsize=8,ncol=2)
    fig.tight_layout()
    fig.savefig(OUT/"heldout_per_class.png",dpi=180)
    plt.close(fig)
    selected = metrics[lock["selected_candidate"]]
    selected_errors = image_errors[lock["selected_candidate"]]
    weakest_class = min(selected["per_class"],key=lambda c:selected["per_class"][c]["dice"])
    zero = [c for c,v in selected["per_class"].items() if v["recall"] == 0]
    lines = ["# Phase II measured research report", "",
             f"Selected using validation only: **{lock['selected_candidate']}**, checkpoint `final/model.pt`.",
             f"Held-out native-resolution macro Dice **{selected['macro_dice']:.6f}**, IoU **{selected['macro_iou']:.6f}** on 1,006 images.","",
             f"Weakest class: **class {weakest_class}**, Dice **{selected['per_class'][weakest_class]['dice']:.6f}**. Completed research evaluation does not establish industrial deployment reliability.","",
             "Phase I historical validation: Dice **0.1182569**, IoU **0.0774453** at 512×128 and threshold 0.5.",
             "These historical scores use different training/evaluation conditions; they do not prove a Phase II improvement.","",
             f"Fresh A evaluation under original resize/threshold on CPU FP32 / PyTorch {original['torch_version']}: Dice {original['macro_dice']:.9f}, IoU {original['macro_iou']:.9f}. A secondary CUDA FP32 check is retained separately.","",
             "## Matched held-out comparison", "",
             "| Candidate | Macro Dice | Macro IoU | Mean latency (ms) | Peak allocated MiB |", "|---|---:|---:|---:|---:|"]
    for name,m in metrics.items():
        lines.append(f"| {name} | {m['macro_dice']:.6f} | {m['macro_iou']:.6f} | {m['latency_ms']['mean']:.2f} | {m['peak_allocated_bytes']/2**20:.1f} |")
    if "E_weighted_patch" in names:
        lines.extend(["","E changes only BCE positive weights relative to C, using native training priors capped at 1,000. Soft Dice, seed, architecture, sampling, optimizer and initial epoch budget remain fixed.",
                      "E was introduced after historical/new validation showed rare-class collapse, before opening test. E-versus-B cannot isolate patching because objective and representation both differ.",
                      "The primary patching comparison is B-versus-C; E-versus-C is the objective experiment. E's objective-specific loss curve is plotted separately."])
        olow,ohigh = objective_bootstrap["percentile_95_interval"]
        lines.append(f"E minus C macro Dice: {objective_bootstrap['delta_E_minus_C']:+.6f}; paired-image 95% interval [{olow:+.6f}, {ohigh:+.6f}].")
    if "F_present_dice_patch" in names:
        plow,phigh = present_bootstrap["percentile_95_interval"]
        lines.extend(["","F retains C's unweighted BCE and changes only the soft-Dice term to average over GT-positive image/class pairs. Dice contributes zero on an all-negative batch; BCE still penalizes all false positives.",
                      "No empty validation/test images or classes are excluded from the reported metrics. The mask applies only inside the training loss. F uses the same native patch sampling and initial training budget as C.",
                      f"F minus C macro Dice: {present_bootstrap['delta_F_minus_C']:+.6f}; paired-image 95% interval [{plow:+.6f}, {phigh:+.6f}]."])
    if representation_present:
        rlow,rhigh = representation_present["percentile_95_interval"]
        lines.extend(["","G is the matching whole-image model with present-class Dice. B/C and G/F form a controlled representation-by-Dice-reduction comparison; E is a separate BCE-weighting experiment.",
                      f"Under present-class Dice, F (patch) minus G (whole) macro Dice: {representation_present['delta_F_patch_minus_G_whole']:+.6f}; paired-image 95% interval [{rlow:+.6f}, {rhigh:+.6f}].",
                      "This separates representation effects from changes to the training objective. Field of view and resizing remain inherent parts of the representation design."])
    if sampling_bootstrap:
        slow,shigh = sampling_bootstrap["percentile_95_interval"]
        lines.extend(["","H retains F's objective, architecture, seed, source-image order and pixel/update budget, changing only training crop selection. It chooses a grid crop containing the rarest GT class present in each positive source image; negative source images remain uniformly cropped.",
                      "Ground truth guides training only. H inference remains exhaustive, with the same 12-patch reference and locked per-class thresholds. Sampling is established methodology, not a newly invented algorithm.",
                      f"H minus F macro Dice: {sampling_bootstrap['delta_H_minus_F']:+.6f}; paired-image 95% interval [{slow:+.6f}, {shigh:+.6f}].",
                      "H versus G changes representation and crop policy; the component comparisons are F/G for representation and H/F for crop selection."])
    if balance_bootstrap:
        blow,bhigh = balance_bootstrap["percentile_95_interval"]
        lines.extend(["","J changes only H's Dice reduction. Each positive image/class Dice loss is weighted by the inverse empirical training positive-pair frequency, then averaged over all B*C pairs. Empty Dice terms remain zero; BCE is unchanged.",
                      "Weights come from H's deterministic training crop exposure logs. J repeats that same crop schedule from the same seeded initialization; no learned H weights, validation labels or test labels enter its class priors.",
                      f"J minus H macro Dice: {balance_bootstrap['delta_J_minus_H']:+.6f}; paired-image 95% interval [{blow:+.6f}, {bhigh:+.6f}].",
                      "This is an ablation of established inverse-frequency reweighting, not a newly invented loss. The reported segmentation metrics remain unweighted four-class pixel-count metrics."])
    if budget_bootstrap:
        klow,khigh = budget_bootstrap["percentile_95_interval"]
        lines.extend(["",f"K extends {parent_label} from its completed three-epoch state, restoring weights, Adam, AMP scaler and RNG. The only changed training setting is the total epoch cap ({extended_config['epochs']}); early-stopping patience remains two. Parent artifacts remain unchanged.",
                      f"K shares its first three epochs with {parent_label}; they were not trained twice. Additional continuation time: {read(OUT/'K_extended_patch/run_info.json')['continuation_only_seconds']:.1f} seconds. The budget table's total time for K includes the shared parent prefix.",
                      "A real-GPU two-update test verified identical losses, weights, optimizer and scaler with versus without an intervening save/reload. The longer budget was motivated by validation underfitting before test opening.",
                      f"K minus its parent macro Dice: {budget_bootstrap['delta_K_minus_parent']:+.6f}; paired-image 95% interval [{klow:+.6f}, {khigh:+.6f}].",
                      "K has a larger compute budget than B/C/E/F/G/H/J. Its score cannot establish a patching-only or loss-only benefit against those other configurations. The three-epoch comparisons remain the controlled component experiments."])
    lines.extend(["","## Actual training budgets","",
                  "| Candidate | Epochs run | Selected epoch | Attempted updates | Successful updates | Run seconds (train + validation) |",
                  "|---|---:|---:|---:|---:|---:|"])
    for label in lock["candidates"]:
        info = read(OUT/label/"run_info.json")
        lines.append(f"| {label} | {info['epochs_completed']} | {info['best_epoch']} | {info.get('attempted_optimizer_steps',info['optimizer_steps'])} | {info['optimizer_steps']} | {info['duration_seconds']:.1f} |")
    lines.append("Checkpoint selection can retain different epochs even with the same total search budget. Per-epoch histories disclose updates up to each selected checkpoint. Phase I used five epochs, batch size 2 and the original larger training pool; its compute/data budget differs.")
    low,high = bootstrap["percentile_95_interval"]
    lines.extend(["",f"C minus B macro Dice: **{bootstrap['delta_C_minus_B']:+.6f}**, paired-image bootstrap 95% interval [{low:+.6f}, {high:+.6f}].",
                  "This interval describes test-image sampling for fixed weights; only one training seed was run.","",
                  "Stride-256 remains edge-aligned, so its last patches overlap. Uniform/Hann/stride ablations share C weights and C's locked thresholds.",
                  "Overlap changes total inference work and field-of-view multiplicity; compute differences are part of the measured tradeoff.","",
                  "## Selected model: all four classes", "", "| Class | Dice | IoU | Precision | Recall | FP pixels | FN pixels |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    def fmt(v):
        return "undefined" if v is None else f"{v:.4f}"
    for c,v in selected["per_class"].items():
        lines.append(f"| {c} | {fmt(v['dice'])} | {fmt(v['iou'])} | {fmt(v['precision'])} | {fmt(v['recall'])} | {v['false_positive_pixels']} | {v['false_negative_pixels']} |")
    lines.extend(["",f"Normal-image false alarms: {selected_errors['normal_false_alarm_images']} / {selected_errors['normal_image_count']} images, counting any predicted pixel of any class as an alarm. No postprocessing filter is applied.",
                  "", "| Class | GT-positive images | Images with zero recovered GT pixels |", "|---|---:|---:|"])
    for c in range(4):
        lines.append(f"| {c+1} | {selected_errors['gt_positive_images_by_class'][c]} | {selected_errors['complete_gt_miss_images_by_class'][c]} |")
    effect = "The measured patch pipeline improves macro Dice over B on these fixed held-out samples." if bootstrap["delta_C_minus_B"] > 0 else "The measured patch pipeline does not improve macro Dice over B on these fixed held-out samples."
    uncertainty = "The paired-image interval crosses zero; sampling uncertainty does not establish a clear directional effect." if low <= 0 <= high else "The paired-image interval excludes zero for these fixed weights; this is not a multi-seed result."
    lines.extend(["",effect,uncertainty])
    lines.extend(["","## Size-aware error analysis","",
                  "Components use 8-connectivity per class. Small <=1,024 pixels; medium <=8,192; large >8,192 at native resolution.",
                  "Object detection requires recovery of >=10% of its GT pixels. It is not an instance-IoU metric.","",
                  "| Size | Class | GT components | Detected components | Object recall | Pixel recall |","|---|---|---:|---:|---:|---:|"])
    for size,classes in selected["performance_by_size"].items():
        for c,v in classes.items():
            lines.append(f"| {size} | {c} | {v['objects']} | {v['detected_objects']} | {fmt(v['object_recall'])} | {fmt(v['pixel_recall'])} |")
    lines.extend(["","## Research contribution and limits","",
                  "The contribution is a controlled resolution-aware segmentation design and measured accuracy/runtime tradeoff in this system. Patching, blending and U-Net are established methods.",
                  "The coarse resized representation and native patch representation are compared; no simultaneous multi-scale feature fusion or new patching algorithm is claimed.",
                  "B/C hold architecture, optimizer, objective, initialization, training input pixel budget and source-image order fixed. Patch field of view and target exposure differ inherently.",
                  "Native geometry, overlapping probability reconstruction, batching and four-class alignment passed tests; masks and predictions remain in original coordinates.",
                  f"Selected classes with zero recall: {', '.join(zero) if zero else 'none'}. Evaluation completion does not imply industrial deployment readiness.",
                  "The initial three-epoch runs and one training seed are a limited compute budget; any longer continuation is separately identified above. Model-selection validation was reused for threshold tuning, so validation is optimistic; held-out test was excluded from both.",
                  "Attempted and skipped AMP updates are logged; any difference in successful updates is disclosed rather than assumed equal. CPU reference A overlapped part of C's training/validation, so training wall times are not an isolated speed benchmark. Final candidate inference timings were measured serially.",
                  "Class 2 has only 20 positive test images. Near duplicates and steel-coil leakage could not be excluded because acquisition-group metadata is unavailable.",
                  "Random patch training can miss a source image's defects. See per-epoch target-exposure records. False-positive/false-negative counts and both successful/failed full-resolution panels are retained.",
                  "`training_target_exposure.csv` records positive sample and pixel counts for every class/epoch. Native patches can present fewer positive samples but similar total foreground pixels at the matched training input-pixel budget.",
                  "Adaptive coarse-to-fine, pretrained encoders and uncontrolled optimizer/augmentation bundles were not introduced.","",
                  "## Evidence and reproduction","",
                  "`comparison.csv` includes all four class metrics. Candidate test folders include per-image errors and native-resolution panels (input / GT / prediction).",
                  "`training_curves.png`, `heldout_per_class.png`, `paired_uncertainty.json`, checkpoint hashes, split hashes and environment/package versions accompany these results.",
                  "`new_image_inference` contains execution on a previously unlabeled Kaggle steel image; this verifies inference, not accuracy without GT.",
                  "Follow `../README.md` for exact Windows commands, and `../SOURCE_AND_CONTRIBUTION_LOG.md` for sources and AI assistance.",
                  "The original files were verified unchanged after the experiments. Reproduction scripts are provided; a second end-to-end training replication was not run."])
    (OUT/"RESEARCH_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    from .verify_evidence import verify as verify_evidence
    verify_evidence()
    write_json(OUT/"STATUS.json",{"phase2_complete":True,"stage":"held-out evaluation and required evidence completed",
               "selected_candidate":lock["selected_candidate"],"checkpoint_sha256":lock["checkpoint_sha256"],
               "heldout_macro_dice":selected["macro_dice"],"heldout_macro_iou":selected["macro_iou"],
               "new_image_inference":True,"all_four_classes_evaluated":True,"limitations_documented":True,
               "industrial_deployment_validated":False,"zero_recall_classes":zero})


if __name__ == "__main__":
    generate()
