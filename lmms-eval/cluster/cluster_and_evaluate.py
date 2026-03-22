"""Cluster-based layer group assignment for cross-image attention masking.

Phase 2: Build per-layer-group clusters from mantis results (benefit-only).
Phase 3: Predict layer group assignments on target datasets and evaluate.

Usage:
  python scripts/cluster_and_evaluate.py \
      --source_dataset mantis \
      --target_datasets muirbench mirb blink \
      --embeddings_dir embeddings \
      --logs_dir logs/qwen2_5_vl-mask-attention-flex-attention-all

This script is CPU-only; it uses pre-computed embeddings and evaluation results.
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lmms_eval.tasks.mirb.utils import eval_multi_choice, eval_open

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LAYER_GROUPS = ["0-3", "4-7", "8-11", "12-15", "16-19", "20-23", "24-27", "28-31", "32-35"]
MODEL_SUBDIR = "Qwen__Qwen2.5-VL-3B-Instruct"
EMBEDDING_TYPES = ["full_seq_avg", "text_full_avg", "text_question_avg", "image_avg", "image_diff"]
DISTANCE_METRICS = ["cosine", "euclidean"]

BLINK_SUBTASKS = [
    "blink_art_style", "blink_counting", "blink_forensic_detection",
    "blink_functional_correspondence", "blink_iq_test", "blink_jigsaw",
    "blink_multi_view_reasoning", "blink_object_localization",
    "blink_relative_depth", "blink_relative_reflectance",
    "blink_semantic_correspondence", "blink_spatial_relation",
    "blink_visual_correspondence", "blink_visual_similarity",
]

# Single-image blink tasks (kept for reference; no longer excluded from aggregation)
BLINK_SINGLE_IMAGE_TASKS = {"blink_counting", "blink_iq_test", "blink_relative_reflectance", "blink_relative_depth"}

# Mantis category mapping (doc_id -> normalized category)
_MANTIS_CAT_CACHE = None

def _mantis_category(doc_id):
    global _MANTIS_CAT_CACHE
    if _MANTIS_CAT_CACHE is None:
        cat_path = os.path.join(os.path.dirname(__file__), "..",
                                "clustering_results", "mantis_categories.json")
        if os.path.exists(cat_path):
            with open(cat_path) as f:
                raw = json.load(f)
            _MANTIS_CAT_CACHE = {int(k): v["category"] for k, v in raw.items()}
        else:
            _MANTIS_CAT_CACHE = {}
    return _MANTIS_CAT_CACHE.get(doc_id, "unknown")


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------

def extract_layer_tag(dir_name):
    """Extract layer tag from a directory name like qwen2_5_vl-3b-mask-attention-0-3-all-flex-attention."""
    base = os.path.basename(dir_name)
    tag = base.split("-mask-attention-")[1].split("-all-flex-attention")[0] if "-mask-attention-" in base else "unknown"
    return tag.lower()


def layer_tag_to_group(tag):
    """Classify a layer tag as baseline, single, group, or other."""
    if tag == "baseline":
        return "baseline"
    parts = tag.split("-")
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        return "group"
    if tag.isdigit():
        return "single"
    return "other"


def find_sample_files(logs_dir, experiment_dir, task_name):
    """Find the *_samples_{task_name}.jsonl file in an experiment directory."""
    model_dir = os.path.join(logs_dir, experiment_dir, MODEL_SUBDIR)
    if not os.path.isdir(model_dir):
        return None
    pattern = os.path.join(model_dir, f"*_samples_{task_name}.jsonl")
    files = glob.glob(pattern)
    return files[0] if files else None


def load_sample_correctness(jsonl_path, task_name):
    """Load per-sample correctness and subtask metadata from a JSONL file.

    Returns:
        correctness: {doc_id: bool}
        metadata: {doc_id: {"subtask": str, ...}}
    """
    correctness = {}
    metadata = {}
    with open(jsonl_path) as f:
        for line in f:
            sample = json.loads(line)
            doc_id = sample["doc_id"]
            meta = {}

            if task_name == "mantis":
                correct = sample["mantis_score"]["correct"]
                qid = sample["mantis_score"].get("question_id", "")
                meta["subtask"] = _mantis_category(doc_id)
                meta["question_id"] = qid
            elif task_name == "muirbench":
                score = sample["muirbench_score_overall"]
                correct = score["pred"].lower().strip() == score["answer"].lower().strip()
                meta["subtask"] = score.get("task", "unknown")
                meta["image_relation"] = score.get("image_relation", "unknown")
                meta["image_type"] = score.get("image_type", "unknown")
            elif task_name == "mirb":
                score = sample["mirb_score"]
                ans = score["answers"]
                pred = score["pred_answer"]
                if ans in ["A", "B", "C", "D", "E"]:
                    correct = eval_multi_choice(ans, pred)
                else:
                    correct = eval_open(ans, pred)
                meta["subtask"] = score.get("subset", "unknown")
            elif task_name.startswith("blink_"):
                correct = sample["blink_acc"]["is_correct"]
                meta["subtask"] = task_name.replace("blink_", "")
            else:
                raise ValueError(f"Unknown task: {task_name}")

            correctness[doc_id] = bool(correct)
            metadata[doc_id] = meta
    return correctness, metadata


def get_task_names_for_dataset(dataset_name):
    """Get the list of task names for a given dataset."""
    if dataset_name == "blink":
        return BLINK_SUBTASKS
    return [dataset_name]


def load_all_results(logs_dir, task_names):
    """Load per-sample correctness for baseline and all layer groups.

    Returns:
        baseline_results: {task_name: {doc_id: bool}}
        layer_group_results: {layer_group: {task_name: {doc_id: bool}}}
        sample_meta: {task_name: {doc_id: {"subtask": str, ...}}}
    """
    baseline_results = {}
    layer_group_results = defaultdict(dict)
    sample_meta = {}

    exp_dirs = [d for d in os.listdir(logs_dir)
                if os.path.isdir(os.path.join(logs_dir, d)) and "3b" in d]

    for exp_dir in exp_dirs:
        tag = extract_layer_tag(exp_dir)
        tag_type = layer_tag_to_group(tag)

        for task_name in task_names:
            jsonl_path = find_sample_files(logs_dir, exp_dir, task_name)
            if jsonl_path is None:
                continue

            correctness, meta = load_sample_correctness(jsonl_path, task_name)

            if tag == "baseline":
                baseline_results[task_name] = correctness
                sample_meta[task_name] = meta
            elif tag_type == "group" and tag in LAYER_GROUPS:
                layer_group_results[tag][task_name] = correctness

    return baseline_results, dict(layer_group_results), sample_meta


# ---------------------------------------------------------------------------
# Embedding loading
# ---------------------------------------------------------------------------

def load_embeddings(embeddings_dir, task_name):
    """Load embeddings for a task.

    Returns: list of dicts with doc_id and embedding tensors
    """
    path = os.path.join(embeddings_dir, task_name, "embeddings.pt")
    if not os.path.exists(path):
        print(f"  WARNING: No embeddings found at {path}")
        return []
    return torch.load(path, map_location="cpu", weights_only=False)


def embeddings_to_matrix(embeddings_list, emb_type):
    """Stack embeddings of a given type into a matrix.

    Returns:
        doc_ids: list of doc_ids
        matrix: (N, hidden_size) numpy array
    """
    doc_ids = []
    vectors = []
    for emb in embeddings_list:
        doc_ids.append(emb["doc_id"])
        vectors.append(emb[emb_type].float().numpy())
    return doc_ids, np.array(vectors, dtype=np.float32)


# ---------------------------------------------------------------------------
# Cluster construction (Phase 2)
# ---------------------------------------------------------------------------

def build_clusters(source_task_names, baseline_results, layer_group_results,
                   source_embeddings_all, emb_type):
    """Build cluster representations for each layer group.

    A sample is in cluster for layer_group G if:
      - It is INCORRECT at baseline
      - It is CORRECT when layer group G is masked

    Returns:
        cluster_reps: {layer_group: numpy array (hidden_size,)}
        cluster_sizes: {layer_group: int}
    """
    cluster_reps = {}
    cluster_sizes = {}

    for lg in LAYER_GROUPS:
        member_vectors = []
        for task_name in source_task_names:
            if task_name not in baseline_results:
                continue
            if task_name not in layer_group_results.get(lg, {}):
                continue

            baseline = baseline_results[task_name]
            masked = layer_group_results[lg][task_name]

            # Load embeddings for this task
            embeddings = source_embeddings_all.get(task_name, [])
            emb_by_doc = {e["doc_id"]: e for e in embeddings}

            for doc_id in baseline:
                if doc_id not in masked or doc_id not in emb_by_doc:
                    continue
                # Benefit-only: incorrect at baseline, correct with masking
                if not baseline[doc_id] and masked[doc_id]:
                    member_vectors.append(emb_by_doc[doc_id][emb_type].float().numpy())

        if member_vectors:
            cluster_reps[lg] = np.mean(member_vectors, axis=0)
            cluster_sizes[lg] = len(member_vectors)
        else:
            cluster_sizes[lg] = 0

    return cluster_reps, cluster_sizes


# ---------------------------------------------------------------------------
# Prediction (Phase 3)
# ---------------------------------------------------------------------------

def compute_similarity(sample_vec, cluster_reps, metric):
    """Compute similarity between a sample and all cluster representatives.

    Returns: {layer_group: similarity_score}
    Higher is better for both metrics (we negate euclidean).
    """
    scores = {}
    for lg, rep in cluster_reps.items():
        if metric == "cosine":
            norm_s = np.linalg.norm(sample_vec)
            norm_r = np.linalg.norm(rep)
            if norm_s > 0 and norm_r > 0:
                scores[lg] = float(np.dot(sample_vec, rep) / (norm_s * norm_r))
            else:
                scores[lg] = 0.0
        elif metric == "euclidean":
            scores[lg] = -float(np.linalg.norm(sample_vec - rep))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    return scores


def predict_and_evaluate(target_task_names, target_embeddings_all,
                         baseline_results, layer_group_results,
                         cluster_reps, emb_type, metric, threshold,
                         macro_avg=False, sample_meta=None,
                         collect_assignments=False):
    """Predict layer group for each target sample and evaluate.

    Args:
        sample_meta: if provided, {task_name: {doc_id: {"subtask": ...}}} used
            for per-sample assignment tracking.
        collect_assignments: if True, return per-sample assignment records in
            the result dict under "assignments".

    Returns dict with:
        accuracy, n_masked, n_baseline, total, per_task,
        and optionally "assignments" (list of per-sample dicts).
    """
    total_correct = 0
    total_samples = 0
    n_masked = 0
    n_baseline_fallback = 0
    per_task = {}
    per_subtask = defaultdict(lambda: {"correct": 0, "total": 0}) if sample_meta else None
    assignments = [] if collect_assignments else None

    for task_name in target_task_names:
        if task_name not in baseline_results:
            continue

        embeddings = target_embeddings_all.get(task_name, [])
        baseline = baseline_results[task_name]

        task_correct = 0
        task_total = 0
        task_masked = 0

        for emb in embeddings:
            doc_id = emb["doc_id"]
            if doc_id not in baseline:
                continue

            sample_vec = emb[emb_type].float().numpy()

            if cluster_reps:
                scores = compute_similarity(sample_vec, cluster_reps, metric)
                best_lg = max(scores, key=scores.get)
                best_score = scores[best_lg]
            else:
                best_score = -float("inf")
                best_lg = None

            if best_score >= threshold and best_lg is not None:
                lg_results = layer_group_results.get(best_lg, {}).get(task_name, {})
                correct = lg_results.get(doc_id, False)
                task_masked += 1
                n_masked += 1
                assigned_lg = best_lg
            else:
                correct = baseline[doc_id]
                n_baseline_fallback += 1
                assigned_lg = "baseline"

            if collect_assignments:
                baseline_correct = baseline[doc_id]
                masked_correct = correct
                if assigned_lg == "baseline":
                    outcome = "unchanged"
                elif masked_correct and not baseline_correct:
                    outcome = "improved"
                elif not masked_correct and baseline_correct:
                    outcome = "degraded"
                else:
                    outcome = "unchanged"

                rec = {
                    "doc_id": doc_id,
                    "task_name": task_name,
                    "assigned_layer_group": assigned_lg,
                    "similarity_score": float(best_score) if best_lg else None,
                    "baseline_correct": bool(baseline_correct),
                    "masked_correct": bool(masked_correct),
                    "outcome": outcome,
                }
                if sample_meta and task_name in sample_meta and doc_id in sample_meta[task_name]:
                    rec.update(sample_meta[task_name][doc_id])
                assignments.append(rec)

            if correct:
                task_correct += 1
            task_total += 1
            total_correct += int(correct)
            total_samples += 1

            if per_subtask is not None and task_name in sample_meta and doc_id in sample_meta[task_name]:
                st = sample_meta[task_name][doc_id].get("subtask", task_name)
                per_subtask[st]["correct"] += int(correct)
                per_subtask[st]["total"] += 1

        if task_total > 0:
            per_task[task_name] = {
                "accuracy": task_correct / task_total,
                "correct": task_correct,
                "total": task_total,
                "masked": task_masked,
            }

    if not per_task:
        accuracy = 0.0
    elif macro_avg:
        accuracy = float(np.mean([v["accuracy"] for v in per_task.values()]))
    else:
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    result = {
        "accuracy": accuracy,
        "n_masked": n_masked,
        "n_baseline": n_baseline_fallback,
        "total": total_samples,
        "per_task": per_task,
    }
    if per_subtask is not None:
        result["per_subtask"] = {
            st: {"accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0.0, **v}
            for st, v in per_subtask.items()
        }
    if collect_assignments:
        result["assignments"] = assignments
    return result


def compute_ceiling(target_task_names, baseline_results, layer_group_results,
                    macro_avg=False):
    """Compute ceiling accuracy: for each sample, pick the best layer group."""
    per_task = {}

    for task_name in target_task_names:
        if task_name not in baseline_results:
            continue

        baseline = baseline_results[task_name]
        task_correct = 0
        task_total = 0

        for doc_id in baseline:
            best = baseline[doc_id]
            for lg in LAYER_GROUPS:
                lg_res = layer_group_results.get(lg, {}).get(task_name, {})
                if lg_res.get(doc_id, False):
                    best = True
                    break
            if best:
                task_correct += 1
            task_total += 1

        if task_total > 0:
            per_task[task_name] = task_correct / task_total

    if not per_task:
        return 0.0, per_task

    if macro_avg:
        accuracy = float(np.mean(list(per_task.values())))
    else:
        total_correct = 0
        total_samples = 0
        for task_name in per_task:
            baseline = baseline_results[task_name]
            n = len(baseline)
            total_correct += int(round(per_task[task_name] * n))
            total_samples += n
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    return accuracy, per_task


def compute_baseline_accuracy(target_task_names, baseline_results, macro_avg=False):
    """Compute baseline accuracy (no masking).

    When macro_avg=True, return the mean of per-task accuracies (used for blink
    to match the official lmms-eval aggregation).
    """
    per_task = {}

    for task_name in target_task_names:
        if task_name not in baseline_results:
            continue
        baseline = baseline_results[task_name]
        correct = sum(baseline.values())
        total = len(baseline)
        per_task[task_name] = correct / total if total > 0 else 0.0

    if not per_task:
        return 0.0, per_task

    if macro_avg:
        accuracy = float(np.mean(list(per_task.values())))
    else:
        total_correct = sum(sum(baseline_results[t].values()) for t in per_task)
        total_samples = sum(len(baseline_results[t]) for t in per_task)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    return accuracy, per_task


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cluster-based layer group prediction")
    parser.add_argument("--source_dataset", type=str, default="mantis",
                        help="Dataset to build clusters from")
    parser.add_argument("--target_datasets", nargs="+",
                        default=["muirbench", "mirb", "blink"],
                        help="Datasets to evaluate on")
    parser.add_argument("--embeddings_dir", type=str,
                        default="/lustre/groups/eml/projects/huang/multi_image/lmms-eval/embeddings")
    parser.add_argument("--logs_dir", type=str,
                        default="/lustre/groups/eml/projects/huang/multi_image/lmms-eval/logs/qwen2_5_vl-mask-attention-flex-attention-all")
    parser.add_argument("--output_dir", type=str,
                        default="/lustre/groups/eml/projects/huang/multi_image/lmms-eval/clustering_results")
    parser.add_argument("--threshold_steps", type=int, default=21,
                        help="Number of threshold steps from 0 to 1")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Gather all task names (include all blink subtasks to match official aggregation)
    source_task_names = get_task_names_for_dataset(args.source_dataset)
    all_target_task_names = []
    for ds in args.target_datasets:
        all_target_task_names.extend(get_task_names_for_dataset(ds))

    all_task_names = list(set(source_task_names + all_target_task_names))

    print("Loading evaluation results...")
    baseline_results, layer_group_results, sample_meta = load_all_results(
        args.logs_dir, all_task_names)

    print(f"\nBaseline results loaded for tasks: {list(baseline_results.keys())}")
    print(f"Layer groups with results: {list(layer_group_results.keys())}")
    for lg in LAYER_GROUPS:
        if lg in layer_group_results:
            print(f"  {lg}: tasks = {list(layer_group_results[lg].keys())}")

    # Load embeddings
    print("\nLoading embeddings...")
    source_embeddings = {}
    for task_name in source_task_names:
        embs = load_embeddings(args.embeddings_dir, task_name)
        if embs:
            source_embeddings[task_name] = embs
            print(f"  Source {task_name}: {len(embs)} samples")

    target_embeddings = {}
    for task_name in all_target_task_names:
        embs = load_embeddings(args.embeddings_dir, task_name)
        if embs:
            target_embeddings[task_name] = embs
            print(f"  Target {task_name}: {len(embs)} samples")

    # Compute baselines and ceilings for each target dataset
    print("\n" + "=" * 80)
    print("REFERENCE METRICS")
    print("=" * 80)

    for ds_name in args.target_datasets:
        ds_tasks = get_task_names_for_dataset(ds_name)
        macro = (ds_name == "blink")

        baseline_acc, baseline_per_task = compute_baseline_accuracy(
            ds_tasks, baseline_results, macro_avg=macro)
        ceiling_acc, ceiling_per_task = compute_ceiling(
            ds_tasks, baseline_results, layer_group_results, macro_avg=macro)

        print(f"\n--- {ds_name} ---")
        print(f"  Baseline accuracy:  {baseline_acc:.4f}")
        print(f"  Ceiling accuracy:   {ceiling_acc:.4f}")
        print(f"  Max possible gain:  {ceiling_acc - baseline_acc:.4f}")

        for task in sorted(baseline_per_task.keys()):
            b = baseline_per_task.get(task, 0)
            c = ceiling_per_task.get(task, 0)
            print(f"    {task:40s}  baseline={b:.4f}  ceiling={c:.4f}  gap={c-b:.4f}")

    # Run ablation: all embedding types x distance metrics x thresholds
    print("\n" + "=" * 80)
    print("ABLATION: embedding_type x distance_metric x threshold")
    print("=" * 80)

    thresholds = np.linspace(0.0, 1.0, args.threshold_steps).tolist()
    # For euclidean, thresholds are negative (since we negate distances)
    # Use a separate range that makes sense for euclidean
    euclidean_thresholds = np.linspace(-100.0, 0.0, args.threshold_steps).tolist()

    all_results = {}

    for emb_type in EMBEDDING_TYPES:
        print(f"\n  Embedding type: {emb_type}")

        # Build clusters from source dataset
        cluster_reps, cluster_sizes = build_clusters(
            source_task_names, baseline_results, layer_group_results,
            source_embeddings, emb_type
        )

        total_members = sum(cluster_sizes.values())
        print(f"    Clusters built: {sum(1 for s in cluster_sizes.values() if s > 0)}/{len(LAYER_GROUPS)} non-empty")
        print(f"    Total benefit-only samples: {total_members}")
        for lg in LAYER_GROUPS:
            print(f"      {lg}: {cluster_sizes.get(lg, 0)} samples")

        for metric in DISTANCE_METRICS:
            key = f"{emb_type}__{metric}"
            thresh_list = thresholds if metric == "cosine" else euclidean_thresholds

            best_acc = -1
            best_threshold = None
            sweep_results = []

            for ds_name in args.target_datasets:
                ds_tasks = get_task_names_for_dataset(ds_name)
                macro = (ds_name == "blink")

                ds_target_embs = {t: target_embeddings[t] for t in ds_tasks if t in target_embeddings}

                for thresh in thresh_list:
                    result = predict_and_evaluate(
                        ds_tasks, ds_target_embs,
                        baseline_results, layer_group_results,
                        cluster_reps, emb_type, metric, thresh,
                        macro_avg=macro, sample_meta=sample_meta
                    )
                    sweep_results.append({
                        "dataset": ds_name,
                        "emb_type": emb_type,
                        "metric": metric,
                        "threshold": thresh,
                        **result,
                    })

                    if result["accuracy"] > best_acc:
                        best_acc = result["accuracy"]
                        best_threshold = thresh

            all_results[key] = sweep_results

            # Print best result
            print(f"    {metric:10s} | best_threshold={best_threshold:.3f} | best_acc={best_acc:.4f}")

    # Summary table: best accuracy per (emb_type, metric) for each target dataset
    print("\n" + "=" * 80)
    print("SUMMARY: Best accuracy per (embedding_type, distance_metric) per dataset")
    print("=" * 80)

    header = f"{'emb_type':25s} {'metric':10s}"
    for ds_name in args.target_datasets:
        header += f" | {ds_name:>12s}"
    print(header)
    print("-" * len(header))

    for emb_type in EMBEDDING_TYPES:
        for metric in DISTANCE_METRICS:
            key = f"{emb_type}__{metric}"
            row = f"{emb_type:25s} {metric:10s}"
            for ds_name in args.target_datasets:
                ds_results = [r for r in all_results[key] if r["dataset"] == ds_name]
                if ds_results:
                    best = max(ds_results, key=lambda r: r["accuracy"])
                    row += f" | {best['accuracy']:12.4f}"
                else:
                    row += f" | {'N/A':>12s}"
            print(row)

    # Print baseline row for comparison
    baseline_row = f"{'baseline':25s} {'---':10s}"
    for ds_name in args.target_datasets:
        ds_tasks = get_task_names_for_dataset(ds_name)
        macro = (ds_name == "blink")
        b_acc, _ = compute_baseline_accuracy(ds_tasks, baseline_results, macro_avg=macro)
        baseline_row += f" | {b_acc:12.4f}"
    print(baseline_row)

    ceiling_row = f"{'ceiling':25s} {'---':10s}"
    for ds_name in args.target_datasets:
        ds_tasks = get_task_names_for_dataset(ds_name)
        macro = (ds_name == "blink")
        c_acc, _ = compute_ceiling(ds_tasks, baseline_results, layer_group_results,
                                   macro_avg=macro)
        ceiling_row += f" | {c_acc:12.4f}"
    print(ceiling_row)

    # Threshold sensitivity: print full sweep for best embedding type
    print("\n" + "=" * 80)
    print("THRESHOLD SENSITIVITY (best embedding type per dataset)")
    print("=" * 80)

    for ds_name in args.target_datasets:
        # Find the best (emb_type, metric) combo for this dataset
        best_key = None
        best_acc = -1
        for key, results in all_results.items():
            ds_results = [r for r in results if r["dataset"] == ds_name]
            for r in ds_results:
                if r["accuracy"] > best_acc:
                    best_acc = r["accuracy"]
                    best_key = key

        if best_key is None:
            continue

        print(f"\n--- {ds_name} (best combo: {best_key}) ---")
        print(f"{'threshold':>12s} {'accuracy':>10s} {'n_masked':>10s} {'n_baseline':>12s} {'total':>8s}")

        ds_results = sorted(
            [r for r in all_results[best_key] if r["dataset"] == ds_name],
            key=lambda r: r["threshold"]
        )
        for r in ds_results:
            print(f"{r['threshold']:12.3f} {r['accuracy']:10.4f} {r['n_masked']:10d} {r['n_baseline']:12d} {r['total']:8d}")

    # -----------------------------------------------------------------------
    # Collect per-sample assignments for the best config per dataset
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("COLLECTING PER-SAMPLE ASSIGNMENTS (best config per dataset)")
    print("=" * 80)

    all_assignments = {}
    for ds_name in args.target_datasets:
        ds_tasks = get_task_names_for_dataset(ds_name)
        macro = (ds_name == "blink")
        ds_target_embs = {t: target_embeddings[t] for t in ds_tasks if t in target_embeddings}

        best_key = None
        best_acc = -1
        best_thresh = None
        for key, results in all_results.items():
            ds_results = [r for r in results if r["dataset"] == ds_name]
            for r in ds_results:
                if r["accuracy"] > best_acc:
                    best_acc = r["accuracy"]
                    best_key = key
                    best_thresh = r["threshold"]

        if best_key is None:
            continue

        best_emb_type, best_metric = best_key.split("__")
        print(f"\n  {ds_name}: best={best_key} threshold={best_thresh:.3f}")

        cluster_reps, cluster_sizes = build_clusters(
            source_task_names, baseline_results, layer_group_results,
            source_embeddings, best_emb_type
        )

        result = predict_and_evaluate(
            ds_tasks, ds_target_embs,
            baseline_results, layer_group_results,
            cluster_reps, best_emb_type, best_metric, best_thresh,
            macro_avg=macro, sample_meta=sample_meta,
            collect_assignments=True
        )

        all_assignments[ds_name] = {
            "config": {
                "emb_type": best_emb_type,
                "metric": best_metric,
                "threshold": best_thresh,
            },
            "accuracy": result["accuracy"],
            "assignments": result["assignments"],
        }
        print(f"    Collected {len(result['assignments'])} sample assignments")

    # Also collect source (mantis) cluster composition
    source_composition = {}
    for emb_type in EMBEDDING_TYPES[:1]:
        for lg in LAYER_GROUPS:
            members = []
            for task_name in source_task_names:
                if task_name not in baseline_results:
                    continue
                if task_name not in layer_group_results.get(lg, {}):
                    continue
                baseline = baseline_results[task_name]
                masked = layer_group_results[lg][task_name]
                for doc_id in baseline:
                    if doc_id not in masked:
                        continue
                    if not baseline[doc_id] and masked[doc_id]:
                        rec = {"doc_id": doc_id, "task_name": task_name}
                        if task_name in sample_meta and doc_id in sample_meta[task_name]:
                            rec.update(sample_meta[task_name][doc_id])
                        members.append(rec)
            source_composition[lg] = members

    assignments_path = os.path.join(args.output_dir, "sample_assignments.json")
    with open(assignments_path, "w") as f:
        json.dump({
            "target_assignments": all_assignments,
            "source_composition": source_composition,
        }, f, indent=2, default=str)
    print(f"\nSample assignments saved to {assignments_path}")

    # Save all results
    output_path = os.path.join(args.output_dir, "clustering_results.json")
    serializable = {}
    for key, results in all_results.items():
        serializable[key] = results
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
