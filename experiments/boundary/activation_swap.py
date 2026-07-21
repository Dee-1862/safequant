# Activation swap control: weight vs activation damage.
import argparse
import gc
from pathlib import Path

import torch

from core.datasets import load_wildjailbreak_prompts
from core.evaluation import _extract_behavior, classify_batch
from core.manifest import RunConfig, save_result
from core.model_loader import load_fp32, load_quantized
from core.residuals import get_inner, prompt_text
from core.seeds import DEFAULT_SEED, set_global_seed

SEED = set_global_seed(DEFAULT_SEED)
_SCRIPT = Path(__file__)


def capture_quant_activations(q_model, q_tok, prompts, layer_idx):
    """
    Capture residual stream activations at one layer for each prompt.

    Inputs:
        - q_model (nn.Module): Quantized model
        - q_tok (PreTrainedTokenizer): Tokenizer matched to q_model
        - prompts (list): Prompt items
        - layer_idx (int): Layer index to hook

    Outputs:
        - captured (list[Tensor]): CPU tensors of residual activations per prompt
    """
    # Hook buffer for the target layer
    inner = get_inner(q_model)
    device = next(q_model.parameters()).device
    captured = []
    buf = {}

    def hook(module, inputs, output):
        """
        Store the residual stream tensor from the hooked layer.

        Inputs:
            - module (nn.Module): Hooked module
            - inputs (tuple): Forward inputs (unused)
            - output (Tensor | tuple): Layer output

        Outputs:
            - None
        """
        # Prefer the hidden-state tensor when output is a tuple
        r = output[0] if isinstance(output, (tuple, list)) else output
        buf["x"] = r.detach()

    # Run each prompt and keep the captured activation on CPU
    h = inner.layers[layer_idx].register_forward_hook(hook)
    try:
        for p in prompts:
            text = prompt_text(p, q_tok)
            inputs = q_tok(text, return_tensors = "pt", truncation = True, max_length = 512).to(device)
            buf.clear()
            with torch.no_grad():
                _ = q_model(**inputs)
            captured.append(buf["x"].cpu())
    finally:
        h.remove()
    return captured


def generate_with_layer_swap(fp32_model, fp32_tok, prompts, layer_idx, quant_acts, max_new_tokens = 256):
    """
    Generate with FP32 weights while injecting quantized activations at one layer.

    Inputs:
        - fp32_model (nn.Module): FP32 model used for generation
        - fp32_tok (PreTrainedTokenizer): Tokenizer matched to fp32_model
        - prompts (list): Prompt items
        - layer_idx (int): Layer index to overwrite on the first forward
        - quant_acts (list[Tensor]): Pre-captured quantized activations
        - max_new_tokens (int): Generation length cap

    Outputs:
        - fp_responses (list[str]): Decoded completions
    """
    # Resolve device and accumulate outputs
    inner = get_inner(fp32_model)
    device = next(fp32_model.parameters()).device
    fp_responses = []

    # One prompt at a time so the swap hook can use the matching activation
    for i, p in enumerate(prompts):
        text = prompt_text(p, fp32_tok)
        inputs = fp32_tok(text, return_tensors = "pt", truncation = True, max_length = 512).to(device)
        prompt_len = inputs.input_ids.shape[1]
        q_act = quant_acts[i].to(device).to(next(fp32_model.parameters()).dtype)
        if q_act.shape[1] != prompt_len:
            q_act = q_act[:, : min(q_act.shape[1], prompt_len), :]
        first_call = [True]

        def hook(module, inputs_, output):
            """
            Replace the first prompt-length residual with the quantized activation.

            Inputs:
                - module (nn.Module): Hooked module
                - inputs_ (tuple): Forward inputs (unused)
                - output (Tensor | tuple): Layer output

            Outputs:
                - output (Tensor | tuple): Swapped or original output
            """
            # Only rewrite the first forward that still covers the prompt
            r = output[0] if isinstance(output, (tuple, list)) else output
            if first_call[0] and r.shape[1] == prompt_len:
                first_call[0] = False
                new_r = q_act.to(r.dtype)
                if new_r.shape == r.shape:
                    if isinstance(output, tuple):
                        return (new_r,) + output[1:]
                    return new_r
            return output

        # Register, generate, then always remove the hook
        h = inner.layers[layer_idx].register_forward_hook(hook)
        try:
            with torch.no_grad():
                gen = fp32_model.generate(
                    **inputs,
                    max_new_tokens = max_new_tokens,
                    do_sample = False,
                    pad_token_id = fp32_tok.eos_token_id,
                )
            fp_responses.append(fp32_tok.decode(gen[0, prompt_len:], skip_special_tokens = True).strip())
        finally:
            h.remove()

    return fp_responses


def baseline_generate(model, tok, prompts, max_new_tokens = 256):
    """
    Greedy-generate completions without activation intervention.

    Inputs:
        - model (nn.Module): Model used for generation
        - tok (PreTrainedTokenizer): Tokenizer matched to model
        - prompts (list): Prompt items
        - max_new_tokens (int): Generation length cap

    Outputs:
        - out (list[str]): Decoded completions
    """
    # Decode one completion per prompt
    out = []
    device = next(model.parameters()).device
    for p in prompts:
        text = prompt_text(p, tok)
        inputs = tok(text, return_tensors = "pt", truncation = True, max_length = 512).to(device)
        prompt_len = inputs.input_ids.shape[1]
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens = max_new_tokens,
                do_sample = False,
                pad_token_id = tok.eos_token_id,
            )
        out.append(tok.decode(gen[0, prompt_len:], skip_special_tokens = True).strip())
    return out


def main():
    # Parse CLI arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required = True)
    ap.add_argument("--quantizer", required = True)
    ap.add_argument("--swap_layers", default = "4,11,16,22,29")
    ap.add_argument("--n_eval", type = int, default = 200)
    ap.add_argument("--max_new_tokens", type = int, default = 256)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    swap_layers = [int(x) for x in args.swap_layers.split(",")]

    # Load eval prompts and behaviors
    items = load_wildjailbreak_prompts(n_samples = args.n_eval, config = "eval")
    behaviors = []
    for item in items:
        behaviors.append(_extract_behavior(item))

    # Capture quantized activations and baseline quantized responses
    q_model, q_tok = load_quantized(args.model, quantizer = args.quantizer)
    quant_acts_by_layer = {}
    for layer_idx in swap_layers:
        quant_acts_by_layer[layer_idx] = capture_quant_activations(q_model, q_tok, items, layer_idx)
    quant_responses = baseline_generate(q_model, q_tok, items, max_new_tokens = args.max_new_tokens)
    del q_model
    gc.collect()
    torch.cuda.empty_cache()

    # FP32 baseline responses
    fp32_model, fp32_tok = load_fp32(args.model)
    fp32_responses = baseline_generate(
        model = fp32_model,
        tok = fp32_tok,
        prompts = items,
        max_new_tokens = args.max_new_tokens,
    )

    # Layer-swap intervention conditions
    all_responses = {"fp32_baseline": fp32_responses, "quant_baseline": quant_responses}
    for layer_idx in swap_layers:
        all_responses[f"swap_q_into_fp32_L{layer_idx}"] = generate_with_layer_swap(
            fp32_model = fp32_model,
            fp32_tok = fp32_tok,
            prompts = items,
            layer_idx = layer_idx,
            quant_acts = quant_acts_by_layer[layer_idx],
            max_new_tokens = args.max_new_tokens,
        )
    del fp32_model
    gc.collect()
    torch.cuda.empty_cache()

    # Classify all conditions in one HarmBench session
    keys = list(all_responses.keys())
    flat_b, flat_r = [], []
    for k in keys:
        flat_b.extend(behaviors)
        flat_r.extend(all_responses[k])
    flags = classify_batch(flat_b, flat_r, device = device)
    n = args.n_eval
    asr = {k: 100.0 * sum(flags[i * n : (i + 1) * n]) / n for i, k in enumerate(keys)}

    # Persist ASR table
    cfg = RunConfig(
        model = args.model,
        quantizer = args.quantizer,
        extra = {"n_eval": args.n_eval, "swap_layers": swap_layers, "seed": SEED},
    )
    data = {
        "fp32_baseline_asr": asr["fp32_baseline"],
        "quant_baseline_asr": asr["quant_baseline"],
        "swap_asr": {f"L{L}": asr[f"swap_q_into_fp32_L{L}"] for L in swap_layers},
    }
    path = save_result(_SCRIPT, cfg, data)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
