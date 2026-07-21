# Build FP-refused / quant-complied displacement pairs for Q-Resafe DPO.
import argparse
import gc
import json
from pathlib import Path

import torch

from core.datasets import load_wildjailbreak_prompts
from core.evaluation import generate_only, classify_batch, _extract_behavior
from core.manifest import RunConfig, save_result
from core.model_loader import load_fp32, load_quantized
from core.seeds import set_global_seed, DEFAULT_SEED

_SCRIPT = Path(__file__)

ADVBENCH_SPLIT = (
    Path(__file__).resolve().parents[2]
    / "refusal_direction/dataset/splits/harmful_train.json"
)


def _advbench(n_prompts):
    """
    Load AdvBench instruction prompts from the local refusal_direction split.

    Inputs:
        - n_prompts (int | None): Cap on prompts, or None for all

    Outputs:
        - items (list[dict]): Prompt dicts with prompt/vanilla fields
    """
    # Read the AdvBench JSON split
    with open(ADVBENCH_SPLIT) as f:
        items = json.load(f)
    prompts = [it["instruction"] for it in items if it.get("instruction")]

    # Optional length cap
    prompts = prompts[:n_prompts] if n_prompts else prompts
    return [{"prompt": p, "vanilla": p} for p in prompts]


def get_items(source, n_prompts, wj_config):
    """
    Load prompt items from WildJailbreak or AdvBench.

    Inputs:
        - source (str): "wildjailbreak" or "advbench"
        - n_prompts (int): Number of prompts to load
        - wj_config (str): WildJailbreak split name when source is wildjailbreak

    Outputs:
        - items (list): Prompt items
    """
    # Dispatch by prompt source
    if source == "wildjailbreak":
        return load_wildjailbreak_prompts(n_samples = n_prompts, config = wj_config)
    if source == "advbench":
        return _advbench(n_prompts)
    raise ValueError(f"unknown prompt_source: {source}")


def _send_prompt(item):
    """
    Extract the user-facing prompt string from a heterogeneous item dict.

    Inputs:
        - item (dict): Prompt item

    Outputs:
        - prompt (str): Prompt text sent to the model
    """
    # Prefer prompt, then goal, then instruction
    return (item.get("prompt") or item.get("goal") or item.get("instruction") or "")


def main():
    # Parse CLI arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default = "llama7b")
    ap.add_argument("--quantizer", default = "aqlm2")
    ap.add_argument("--prompt_source", default = "wildjailbreak",
                    choices = ["wildjailbreak", "advbench"])
    ap.add_argument("--wj_config", default = "eval", choices = ["train", "eval"])
    ap.add_argument("--n_prompts", type = int, default = 400)
    ap.add_argument("--max_new_tokens", type = int, default = 256)
    ap.add_argument("--output", default = None,
                    help = "override output path (default: outputs/trained_repair/qresafe_pairs_<model>_<quant>[_<tag>].json)")
    args = ap.parse_args()

    set_global_seed(DEFAULT_SEED)

    # Variant tag for non-default WildJailbreak splits
    pair_tag = args.wj_config if args.prompt_source == "wildjailbreak" and args.wj_config != "train" else None

    # Load prompts and extract send-strings / behaviors
    items = get_items(args.prompt_source, args.n_prompts, args.wj_config)
    sends = [_send_prompt(it) for it in items]
    behaviors = [_extract_behavior(it) for it in items]
    n = len(items)
    print(f"[data] {n} prompts from {args.prompt_source} (config={args.wj_config})")

    # FP completions (yw candidates)
    fp_model, tok = load_fp32(args.model)
    fp_resp = generate_only(fp_model, tok, items, max_new_tokens = args.max_new_tokens,
                            desc = "FP")
    del fp_model
    gc.collect()
    torch.cuda.empty_cache()

    # Quantized completions (yl candidates)
    q_model, _ = load_quantized(args.model, quantizer = args.quantizer)
    q_resp = generate_only(q_model, tok, items, max_new_tokens = args.max_new_tokens,
                           desc = args.quantizer)
    del q_model
    gc.collect()
    torch.cuda.empty_cache()

    # Judge both with HarmBench in one session
    flags = classify_batch(behaviors + behaviors, fp_resp + q_resp, device = "cuda")
    fp_harmful = flags[:n]
    q_harmful = flags[n:]
    fp_asr = 100.0 * sum(fp_harmful) / n
    q_asr = 100.0 * sum(q_harmful) / n
    print(f"[asr] FP={fp_asr:.1f}%  {args.quantizer}={q_asr:.1f}%  "
          f"({args.prompt_source}, HarmBench judge)")

    # Displacement set: FP refused, quantized complied
    pairs = []
    for sp, fw, fh, qr, qh in zip(sends, fp_resp, fp_harmful, q_resp, q_harmful):
        if (not fh) and qh:
            pairs.append({"instruction": sp, "yw": fw.strip(), "yl": qr.strip()})
    print(f"[pairs] {len(pairs)} displacement pairs (FP refused & {args.quantizer} complied)")
    if len(pairs) < 20:
        print(f"  WARNING: only {len(pairs)} displacement pairs -- saliency may be noisy. "
              f"Consider widening n_prompts or relaxing the filter.")

    # Persist pairs JSON
    cfg = RunConfig(
        model = args.model,
        quantizer = args.quantizer,
        variant = pair_tag,
        extra = {
            "prompt_source": args.prompt_source,
            "wj_config": args.wj_config,
            "n_prompts": n,
            "fp_asr": fp_asr,
            "q_asr": q_asr,
            "max_new_tokens": args.max_new_tokens,
            "filter": "fp_refused AND quant_complied",
        },
    )
    data = {"n_pairs": len(pairs), "pairs": pairs}
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents = True, exist_ok = True)
        out_path.write_text(json.dumps({"meta": cfg.to_dict(), **data}, indent = 2))
    else:
        out_path = save_result(_SCRIPT, cfg, data)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
