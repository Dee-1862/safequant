# Residual-stream extraction and model unwrapping (single canonical impl).

from __future__ import annotations

import torch


def get_inner(model):
    """
    Unwrap to the inner model that has .layers (handles GPTQ wrappers).

    Inputs:
        - model (torch.nn.Module): Possibly wrapped causal LM

    Outputs:
        - inner (torch.nn.Module): Module exposing .layers
    """
    # Peel one .model if present
    inner = model.model if hasattr(model, "model") else model

    # Some wrappers nest another .model that owns .layers
    if hasattr(inner, "model") and hasattr(inner.model, "layers"):
        inner = inner.model
    return inner


def prompt_text(p, tok):
    """
    Format a prompt (str or dict) with the model chat template.

    Inputs:
        - p (str | dict): Raw prompt or dict with prompt/vanilla/behavior
        - tok (Tokenizer): Tokenizer with apply_chat_template

    Outputs:
        - text (str): Formatted prompt string
    """
    # Normalize dict prompts to a single user string
    if isinstance(p, dict):
        text = p.get("prompt") or p.get("vanilla") or p.get("behavior") or ""
    else:
        text = str(p)

    # Prefer chat template when the tokenizer supports it
    try:
        return tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize = False,
            add_generation_prompt = True,
        )
    except Exception:
        return text


def default_token_pos(model_name):
    """
    Per-model heuristic for post-instruction residual token position.

    Inputs:
        - model_name (str): Model alias or HF id

    Outputs:
        - token_pos (int): Relative token index into the residual sequence
    """
    # Lowercase for substring checks
    name = model_name.lower()

    # Llama / Phi use different post-instruction offsets; others use last token
    if "llama" in name:
        return -5
    if "phi" in name:
        return -3
    return -1


def extract_last_token_residuals(
    model, tokenizer, prompts, layers, device, token_pos = -1
):
    """
    Collect per-layer last-token residuals for each prompt.

    Inputs:
        - model (torch.nn.Module): Causal LM
        - tokenizer (Tokenizer): Matching tokenizer
        - prompts (list): Prompts (str or dict)
        - layers (list): Layer indices to hook
        - device (torch.device): Device for tokenization / forward
        - token_pos (int): Relative token index into each sequence

    Outputs:
        - out (dict): layer -> list of CPU float tensors (one per prompt)
    """
    # Hooks write into this buffer each forward pass
    inner = get_inner(model)
    buf = {}

    def make_hook(layer_idx):
        """
        Build a hook that stores the selected token residual for one layer.

        Inputs:
            - layer_idx (int): Layer index key into buf

        Outputs:
            - hook (callable): Forward hook
        """
        def hook(module, inputs, output):
            """
            Capture residual at token_pos into buf[layer_idx].

            Inputs:
                - module (nn.Module): Hooked module (unused)
                - inputs (tuple): Forward inputs (unused)
                - output (Tensor | tuple): Layer output

            Outputs:
                - None
            """
            r = output[0] if isinstance(output, (tuple, list)) else output
            if r.dim() == 3:
                buf[layer_idx] = r[:, token_pos, :].detach().cpu().float().squeeze(0)

        return hook

    # Register hooks, then run one forward per prompt
    handles = []
    for l in layers:
        handles.append(inner.layers[l].register_forward_hook(make_hook(l)))
    out = {l: [] for l in layers}
    try:
        for p in prompts:
            text = prompt_text(p, tokenizer)
            inputs = tokenizer(
                text, return_tensors = "pt", truncation = True, max_length = 512
            ).to(device)
            buf.clear()
            with torch.no_grad():
                _ = model(**inputs)
            for l, r in buf.items():
                out[l].append(r)
    finally:
        for h in handles:
            h.remove()
    return out
