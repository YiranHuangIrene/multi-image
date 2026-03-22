"""Extract input_embeds from Qwen2.5-VL-3B for clustering experiments.

Loads each benchmark dataset, constructs prompts identically to the lmms_eval
evaluation pipeline, runs a single forward pass through the vision encoder +
projector + text embedding layer, and saves per-sample embeddings of 5 types:
  full_seq_avg, text_full_avg, text_question_avg, image_avg, image_diff

Usage:
  python scripts/extract_embeddings.py --datasets mantis muirbench mirb blink
  python scripts/extract_embeddings.py --verify --datasets mantis   # verification mode
"""

import argparse
import base64
import os
import sys
from io import BytesIO
from itertools import combinations
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qwen_vl_utils import process_vision_info

from lmms_eval.tasks.blink.utils import blink_doc_to_text, blink_doc_to_visual
from lmms_eval.tasks.mantis.utils import mantis_doc_to_text, mantis_doc_to_visual
from lmms_eval.tasks.mirb.utils import mirb_doc_to_text, mirb_doc_to_visual
from lmms_eval.tasks.muirbench.utils import muir_doc_to_text, muir_doc_to_visual

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
MAX_PIXELS = 12845056
MIN_PIXELS = 256 * 28 * 28
SYSTEM_PROMPT = "You are a helpful assistant."
HIDDEN_SIZE = 2048

BLINK_SUBTASKS = {
    "blink_art_style": "Art_Style",
    "blink_counting": "Counting",
    "blink_forensic_detection": "Forensic_Detection",
    "blink_functional_correspondence": "Functional_Correspondence",
    "blink_iq_test": "IQ_Test",
    "blink_jigsaw": "Jigsaw",
    "blink_multi_view_reasoning": "Multi-view_Reasoning",
    "blink_object_localization": "Object_Localization",
    "blink_relative_depth": "Relative_Depth",
    "blink_relative_reflectance": "Relative_Reflectance",
    "blink_semantic_correspondence": "Semantic_Correspondence",
    "blink_spatial_relation": "Spatial_Relation",
    "blink_visual_correspondence": "Visual_Correspondence",
    "blink_visual_similarity": "Visual_Similarity",
}

LMMS_KWARGS = {
    "mantis": {"pre_prompt": "", "post_prompt": ""},
    "muirbench": {"pre_prompt": "", "post_prompt": "\nAnswer with the option's letter from the given choices directly."},
    "mirb": {"pre_prompt": "", "post_prompt": ""},
}
BLINK_LMMS_KWARGS = {
    "pre_prompt": "Note: You only need to respond with {} without providing any additional information.\n",
    "post_prompt": "",
}


# ---------------------------------------------------------------------------
# Instruction boundaries per dataset (for text_question_avg)
# ---------------------------------------------------------------------------

def get_instruction_regions(dataset_name, doc):
    """Return the instruction substrings to remove from the user text.

    Returns a dict with optional keys:
      "prefix": instruction at the start of user text
      "suffix": instruction at the end of user text
      "middle": instruction in the middle of user text (searched by substring match)
    Any combination of these can be present.
    Returns empty dict if no instruction should be stripped.
    """
    if dataset_name == "mantis":
        if doc["question_type"] == "short-answer":
            return {
                "prefix": 'Given the images, answer the following short answer vqa question:\nQ: ',
                "suffix": '\nYou can first give your analysis, and then give your final answer as "Final Answer:"',
            }
        else:
            return {
                "middle": "\nAnswer with the option's letter from the given choices directly.",
            }
    elif dataset_name == "muirbench":
        return {"suffix": "\nAnswer with the option's letter from the given choices directly."}
    elif dataset_name == "mirb":
        from lmms_eval.tasks.mirb.utils import get_task_instruction
        instr = get_task_instruction(doc["subset"])
        return {"prefix": instr}
    elif dataset_name.startswith("blink_"):
        num_choices = len(doc["choices"])
        choice_letters = ", ".join([chr(65 + i) for i in range(num_choices)])
        return {"prefix": f"Note: You only need to respond with {choice_letters} without providing any additional information.\n"}
    return {}


# ---------------------------------------------------------------------------
# Prompt construction (mirrors the simple Qwen2.5-VL model in lmms_eval)
# ---------------------------------------------------------------------------

def build_messages(text, images):
    """Build HF-style chat messages from text and PIL images.

    Replicates the message construction in lmms_eval/models/simple/qwen2_5_vl.py.
    """
    processed_visuals = []
    for img in images:
        rgb = img.convert("RGB")
        buf = BytesIO()
        rgb.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        processed_visuals.append({
            "type": "image",
            "image": f"data:image/jpeg;base64,{b64}",
            "max_pixels": MAX_PIXELS,
            "min_pixels": MIN_PIXELS,
        })

    message = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": processed_visuals + [{"type": "text", "text": text}]},
    ]
    return message


# ---------------------------------------------------------------------------
# Token classification
# ---------------------------------------------------------------------------

def classify_tokens(input_ids, tokenizer):
    """Classify each token position into exactly one category.

    Returns a dict mapping category -> list of token indices.
    Categories: system_prompt, template, image_groups (per image), user_text

    Token sequence structure for Qwen2.5-VL chat:
      <|im_start|>system\\n{SYSTEM_PROMPT}<|im_end|>\\n
      <|im_start|>user\\n<|vision_start|>...<|vision_end|>...{text}<|im_end|>\\n
      <|im_start|>assistant\\n
    """
    ids = input_ids.tolist()
    seq_len = len(ids)

    vision_start_id = 151652
    vision_end_id = 151653
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    # Assign each position exactly one label
    labels = ["template"] * seq_len
    image_groups = {}
    current_image_idx = -1

    # State: which block we're currently in
    state = "outside"  # outside | system | user | assistant | image
    state_before_image = "outside"

    for i in range(seq_len):
        tid = ids[i]

        if tid == im_start_id:
            labels[i] = "template"
            # Peek to detect role
            if i + 1 < seq_len:
                role = tokenizer.decode([ids[i + 1]]).strip()
                if role in ("system", "user", "assistant"):
                    state = role
            continue

        if tid == im_end_id:
            labels[i] = "template"
            state = "outside"
            continue

        if tid == vision_start_id:
            labels[i] = "template"
            current_image_idx += 1
            image_groups[current_image_idx] = []
            state_before_image = state
            state = "image"
            continue

        if tid == vision_end_id:
            labels[i] = "template"
            state = state_before_image
            continue

        # Detect role tokens and newlines right after <|im_start|>
        if i >= 1 and ids[i - 1] == im_start_id:
            role = tokenizer.decode([tid]).strip()
            if role in ("system", "user", "assistant"):
                labels[i] = "template"
                continue

        # Newline right after a role token (position i-1 was role, i-2 was im_start)
        if i >= 2 and ids[i - 2] == im_start_id:
            prev_role = tokenizer.decode([ids[i - 1]]).strip()
            if prev_role in ("system", "user", "assistant"):
                decoded = tokenizer.decode([tid])
                if decoded.strip() == "" and "\n" in decoded:
                    labels[i] = "template"
                    continue

        # Newline right after <|im_end|>
        if i >= 1 and ids[i - 1] == im_end_id:
            decoded = tokenizer.decode([tid])
            if decoded.strip() == "" and "\n" in decoded:
                labels[i] = "template"
                continue

        # Classify based on current state
        if state == "system":
            labels[i] = "system_prompt"
        elif state == "user":
            labels[i] = "user_text"
        elif state == "image":
            labels[i] = f"image_{current_image_idx}"
            image_groups[current_image_idx].append(i)
        else:
            labels[i] = "template"

    system_indices = [i for i, l in enumerate(labels) if l == "system_prompt"]
    template_indices = [i for i, l in enumerate(labels) if l == "template"]
    user_text_indices = [i for i, l in enumerate(labels) if l == "user_text"]

    return {
        "system_prompt": system_indices,
        "template": template_indices,
        "image_groups": {k: v for k, v in image_groups.items()},
        "user_text": user_text_indices,
    }


def get_question_only_indices(user_text_indices, instruction_regions, input_ids, tokenizer):
    """Remove instruction tokens from user_text_indices using character-level matching.

    Decodes the actual user text tokens, finds instruction substrings by character
    position, then uses offset mapping from re-encoding to identify which tokens
    to remove. This avoids BPE boundary issues from separate tokenization.
    """
    if not instruction_regions or not user_text_indices:
        return user_text_indices

    ids = input_ids.tolist()
    user_token_ids = [ids[i] for i in user_text_indices]

    full_text = tokenizer.decode(user_token_ids)
    encoding = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoding.offset_mapping

    n_reencoded = len(encoding.input_ids)
    n_original = len(user_text_indices)
    if n_reencoded != n_original:
        print(f"  WARNING: re-encode token count mismatch ({n_reencoded} vs {n_original}), "
              "falling back to full user text")
        return user_text_indices

    # Build set of character positions to remove
    remove_chars = set()

    if "prefix" in instruction_regions:
        prefix = instruction_regions["prefix"]
        if full_text.startswith(prefix):
            remove_chars.update(range(0, len(prefix)))

    if "suffix" in instruction_regions:
        suffix = instruction_regions["suffix"]
        if full_text.endswith(suffix):
            remove_chars.update(range(len(full_text) - len(suffix), len(full_text)))

    if "middle" in instruction_regions:
        middle = instruction_regions["middle"]
        idx = full_text.find(middle)
        if idx >= 0:
            remove_chars.update(range(idx, idx + len(middle)))

    if not remove_chars:
        return user_text_indices

    # Keep tokens where the majority of their characters are outside removal regions.
    # This preserves boundary tokens that straddle instruction/question joins
    # (e.g., "?\n" or ".How") when most of their content is question text.
    keep = []
    for i, (cs, ce) in enumerate(offsets):
        if i >= n_original:
            break
        token_len = ce - cs
        if token_len == 0:
            continue
        overlap = len(set(range(cs, ce)) & remove_chars)
        if overlap <= token_len / 2:
            keep.append(user_text_indices[i])
    return keep


# ---------------------------------------------------------------------------
# Embedding computation
# ---------------------------------------------------------------------------

def compute_embeddings(inputs_embeds, token_classification, question_only_indices):
    """Compute 5 embedding types from inputs_embeds and token classification.

    Args:
        inputs_embeds: (1, seq_len, hidden_size) tensor
        token_classification: dict from classify_tokens
        question_only_indices: list of indices for question-only text tokens

    Returns:
        dict of embedding tensors (each 1D, hidden_size)
    """
    embeds = inputs_embeds.squeeze(0)  # (seq_len, hidden_size)

    # 1. full_seq_avg
    full_seq_avg = embeds.mean(dim=0)

    # 2. text_full_avg
    text_idx = token_classification["user_text"]
    if text_idx:
        text_full_avg = embeds[text_idx].mean(dim=0)
    else:
        text_full_avg = torch.zeros(embeds.shape[-1], device=embeds.device)

    # 3. text_question_avg
    if question_only_indices:
        text_question_avg = embeds[question_only_indices].mean(dim=0)
    else:
        text_question_avg = text_full_avg.clone()

    # 4. image_avg (all image tokens)
    all_image_idx = []
    for group_idx in sorted(token_classification["image_groups"].keys()):
        all_image_idx.extend(token_classification["image_groups"][group_idx])
    if all_image_idx:
        image_avg = embeds[all_image_idx].mean(dim=0)
    else:
        image_avg = torch.zeros(embeds.shape[-1], device=embeds.device)

    # 5. image_diff
    image_groups = token_classification["image_groups"]
    n_images = len(image_groups)
    if n_images >= 2:
        per_image_avgs = []
        for group_idx in sorted(image_groups.keys()):
            idx = image_groups[group_idx]
            if idx:
                per_image_avgs.append(embeds[idx].mean(dim=0))
        if len(per_image_avgs) == 2:
            image_diff = per_image_avgs[1] - per_image_avgs[0]
        else:
            diffs = []
            for i, j in combinations(range(len(per_image_avgs)), 2):
                diffs.append(per_image_avgs[j] - per_image_avgs[i])
            image_diff = torch.stack(diffs).mean(dim=0)
    else:
        image_diff = torch.zeros(embeds.shape[-1], device=embeds.device)

    return {
        "full_seq_avg": full_seq_avg.cpu(),
        "text_full_avg": text_full_avg.cpu(),
        "text_question_avg": text_question_avg.cpu(),
        "image_avg": image_avg.cpu(),
        "image_diff": image_diff.cpu(),
    }


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_benchmark_dataset(dataset_name):
    """Load a benchmark dataset and return (samples_iter, task_name_for_sample).

    For blink subtasks, yields (subtask_name, dataset) pairs.
    For others, yields a single (name, dataset) pair.
    """
    if dataset_name == "mantis":
        ds = load_dataset("TIGER-Lab/Mantis-Eval", split="test", token=True)
        return [("mantis", ds)]
    elif dataset_name == "muirbench":
        ds = load_dataset("MUIRBENCH/MUIRBENCH", split="test", token=True)
        return [("muirbench", ds)]
    elif dataset_name == "mirb":
        ds = load_dataset("VLLMs/MIRB-hf", split="test", token=True)
        return [("mirb", ds)]
    elif dataset_name == "blink":
        result = []
        for subtask_name, hf_config in BLINK_SUBTASKS.items():
            ds = load_dataset("BLINK-Benchmark/BLINK", name=hf_config, split="val")
            result.append((subtask_name, ds))
        return result
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_text_and_images(task_name, doc):
    """Get prompt text and images for a sample, using the same functions as lmms_eval."""
    if task_name == "mantis":
        text = mantis_doc_to_text(doc, lmms_eval_specific_kwargs=LMMS_KWARGS["mantis"])
        images = mantis_doc_to_visual(doc)
    elif task_name == "muirbench":
        text = muir_doc_to_text(doc, lmms_eval_specific_kwargs=LMMS_KWARGS["muirbench"])
        images = muir_doc_to_visual(doc)
    elif task_name == "mirb":
        text = mirb_doc_to_text(doc, lmms_eval_specific_kwargs=LMMS_KWARGS["mirb"])
        images = mirb_doc_to_visual(doc)
    elif task_name.startswith("blink_"):
        text = blink_doc_to_text(doc, lmms_eval_specific_kwargs=BLINK_LMMS_KWARGS)
        images = blink_doc_to_visual(doc)
    else:
        raise ValueError(f"Unknown task: {task_name}")
    return text, images


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_token_classification(input_ids, token_classification, tokenizer, task_name, doc_id):
    """Print detailed token classification breakdown for manual verification."""
    ids = input_ids.tolist()

    print(f"\n{'='*80}")
    print(f"VERIFICATION: task={task_name}, doc_id={doc_id}")
    print(f"Total tokens: {len(ids)}")
    print(f"System prompt tokens: {len(token_classification['system_prompt'])}")
    print(f"Template tokens: {len(token_classification['template'])}")
    for gid, indices in token_classification["image_groups"].items():
        print(f"Image group {gid} tokens: {len(indices)}")
    print(f"User text tokens: {len(token_classification['user_text'])}")

    # Decode each region
    if token_classification["system_prompt"]:
        sys_text = tokenizer.decode([ids[i] for i in token_classification["system_prompt"]])
        print(f"\n--- System prompt text ---\n{repr(sys_text)}")

    if token_classification["user_text"]:
        user_text = tokenizer.decode([ids[i] for i in token_classification["user_text"]])
        print(f"\n--- User text ---\n{repr(user_text)}")

    # Verify no token is unclassified
    classified = set(token_classification["system_prompt"]) | set(token_classification["template"])
    for indices in token_classification["image_groups"].values():
        classified |= set(indices)
    classified |= set(token_classification["user_text"])
    unclassified = set(range(len(ids))) - classified
    if unclassified:
        unclassified_text = tokenizer.decode([ids[i] for i in sorted(unclassified)])
        print(f"\n--- UNCLASSIFIED tokens ({len(unclassified)}) ---")
        print(f"Positions: {sorted(unclassified)}")
        print(f"Text: {repr(unclassified_text)}")
    else:
        print("\nAll tokens classified.")
    print(f"{'='*80}")


def verify_instruction_extraction(task_name, doc, user_text_indices, question_only_indices, input_ids, tokenizer):
    """Print instruction boundary verification for a sample."""
    ids = input_ids.tolist()
    if user_text_indices:
        full_user = tokenizer.decode([ids[i] for i in user_text_indices])
    else:
        full_user = ""
    if question_only_indices:
        question_only = tokenizer.decode([ids[i] for i in question_only_indices])
    else:
        question_only = ""
    stripped_count = len(user_text_indices) - len(question_only_indices)
    regions = get_instruction_regions(task_name, doc)

    print(f"\n--- Instruction extraction: task={task_name} ---")
    if task_name == "mantis":
        print(f"  question_type: {doc.get('question_type', 'N/A')}")
    print(f"  Instruction regions: { {k: repr(v[:80]) for k, v in regions.items()} }")
    print(f"  Full user text ({len(user_text_indices)} tokens): {repr(full_user[:300])}")
    print(f"  Question-only  ({len(question_only_indices)} tokens): {repr(question_only[:300])}")
    print(f"  Stripped tokens: {stripped_count}")


def _get_verify_sample_ids(task_name, ds, n=3):
    """Pick sample IDs for verification, ensuring coverage of different question types."""
    if task_name == "mantis":
        mc_ids = [i for i in range(len(ds)) if ds[i]["question_type"] == "multi-choice"]
        sa_ids = [i for i in range(len(ds)) if ds[i]["question_type"] == "short-answer"]
        picked = mc_ids[:n] + sa_ids[:n]
        return picked if picked else list(range(min(n, len(ds))))
    return list(range(min(n, len(ds))))


def verify_prompt_equivalence(text, messages, processor, tokenizer):
    """Print the full prompt text for manual comparison with lmms_eval output."""
    prompt_text = processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
    if isinstance(prompt_text, list):
        prompt_text = prompt_text[0]
    print(f"\n--- Full prompt text (for comparison with lmms_eval) ---")
    print(prompt_text)
    print(f"--- End of prompt ---\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract input_embeds from Qwen2.5-VL-3B")
    parser.add_argument("--datasets", nargs="+", default=["mantis", "muirbench", "mirb", "blink"],
                        choices=["mantis", "muirbench", "mirb", "blink"],
                        help="Datasets to process")
    parser.add_argument("--output_dir", type=str,
                        default="/lustre/groups/eml/projects/huang/multi_image/lmms-eval/embeddings",
                        help="Directory to save embeddings")
    parser.add_argument("--verify", action="store_true",
                        help="Run in verification mode (process only first 3 samples per dataset)")
    parser.add_argument("--compare_prompt", action="store_true",
                        help="Print full prompts for comparison with lmms_eval")
    parser.add_argument("--model_path", type=str, default=MODEL_NAME,
                        help="HuggingFace model name or local path")
    parser.add_argument("--max_pixels", type=int, default=MAX_PIXELS)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path, max_pixels=args.max_pixels, min_pixels=MIN_PIXELS)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    ).eval()

    # Hook to capture inputs_embeds from the language model
    captured = {}

    def capture_hook(module, args_tuple, kwargs):
        embeds = kwargs.get("inputs_embeds", args_tuple[4] if len(args_tuple) > 4 else None)
        if embeds is not None:
            captured["inputs_embeds"] = embeds.detach()
        return args_tuple, kwargs

    hook_handle = model.model.language_model.register_forward_pre_hook(capture_hook, with_kwargs=True)

    for dataset_name in args.datasets:
        print(f"\n{'#'*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'#'*60}")

        task_datasets = load_benchmark_dataset(dataset_name)

        for task_name, ds in task_datasets:
            print(f"\n  Task: {task_name} ({len(ds)} samples)")
            task_output_dir = os.path.join(args.output_dir, task_name)
            os.makedirs(task_output_dir, exist_ok=True)

            if args.verify:
                sample_ids = _get_verify_sample_ids(task_name, ds, n=3)
            else:
                sample_ids = list(range(len(ds)))
            all_embeddings = []

            for doc_id in tqdm(sample_ids, desc=f"  {task_name}"):
                doc = ds[doc_id]
                text, images = get_text_and_images(task_name, doc)

                if not images:
                    print(f"    Skipping doc_id={doc_id}: no images")
                    continue

                messages = build_messages(text, images)

                if args.compare_prompt:
                    verify_prompt_equivalence(text, messages, processor, tokenizer)

                # Process through the pipeline (same as lmms_eval)
                prompt_text = processor.apply_chat_template(
                    [messages], tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info([messages])
                inputs = processor(
                    text=prompt_text,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    padding_side="right",
                    return_tensors="pt",
                )
                inputs = inputs.to(args.device)
                input_ids = inputs.input_ids[0]

                # Forward pass to capture inputs_embeds
                captured.clear()
                with torch.no_grad():
                    try:
                        model(**inputs)
                    except Exception:
                        # We only need the hook to fire; ignore generation errors
                        pass

                if "inputs_embeds" not in captured:
                    print(f"    WARNING: Failed to capture inputs_embeds for doc_id={doc_id}")
                    continue

                inputs_embeds = captured["inputs_embeds"]

                # Classify tokens
                token_cls = classify_tokens(input_ids, tokenizer)

                if args.verify:
                    verify_token_classification(input_ids, token_cls, tokenizer, task_name, doc_id)

                # Get question-only indices
                instr_regions = get_instruction_regions(task_name, doc)
                question_indices = get_question_only_indices(
                    token_cls["user_text"], instr_regions, input_ids, tokenizer
                )

                if args.verify:
                    verify_instruction_extraction(
                        task_name, doc, token_cls["user_text"],
                        question_indices, input_ids, tokenizer
                    )

                # Compute embeddings
                emb_dict = compute_embeddings(inputs_embeds, token_cls, question_indices)
                emb_dict["doc_id"] = doc_id
                emb_dict["task_name"] = task_name
                emb_dict["n_images"] = len(token_cls["image_groups"])
                all_embeddings.append(emb_dict)

            if not args.verify and all_embeddings:
                out_path = os.path.join(task_output_dir, "embeddings.pt")
                torch.save(all_embeddings, out_path)
                print(f"  Saved {len(all_embeddings)} embeddings to {out_path}")

    hook_handle.remove()
    print("\nDone.")


if __name__ == "__main__":
    main()
