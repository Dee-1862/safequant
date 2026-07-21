# Q-Resafe DPO train + fold: masked LoRA on a quantized model.

# Front Matter
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from core.datasets import load_wildjailbreak_prompts, load_arc_challenge, load_xstest
from core.evaluation import generate_only, classify_batch, _extract_behavior, eval_xstest
from core.manifest import RunConfig, find_result, result_path, save_result
from core.model_loader import load_quantized
from core.restoration import measure_arc
from core.seeds import set_global_seed, DEFAULT_SEED
from experiments.trained_repair.qresafe_fold import quant_cfg, fold_and_writeback_, dequantize_module
from experiments.trained_repair.qresafe_repair import attach_masked_lora, set_lora_enabled, unwrap_lora

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None


def _tokenize_pair(tok, prompt, completion, device):
    """
    Build chat-templated input ids for one (prompt, completion) pair.

    Inputs:
        - tok (PreTrainedTokenizer): Tokenizer with chat template
        - prompt (str): User instruction
        - completion (str): Assistant completion
        - device (str | torch.device): Target device

    Outputs:
        - input_ids (Tensor): Batch-1 token ids
        - n_prompt (int): Number of prompt tokens before the completion
    """
    # Concatenate chat-templated prompt with raw completion tokens
    msgs = [{"role": "user", "content": prompt}]
    p_ids = tok.apply_chat_template(msgs, add_generation_prompt = True, return_tensors = "pt")[0]
    c_ids = tok(completion, add_special_tokens = False, return_tensors = "pt").input_ids[0]
    input_ids = torch.cat([p_ids, c_ids]).unsqueeze(0).to(device)
    n_prompt = p_ids.shape[0]
    return input_ids, n_prompt


def seq_logp(model, input_ids, n_prompt):
    """
    Sum of completion token log-probs (positions >= n_prompt).

    Inputs:
        - model (nn.Module): Model scoring the sequence
        - input_ids (Tensor): Batch-1 token ids
        - n_prompt (int): Prompt length in tokens

    Outputs:
        - logp (Tensor): Scalar sum of completion log-probs
    """
    # Teacher-force log-probs and sum only completion positions
    logits = model(input_ids = input_ids).logits[0]            # [T, V]
    logp = F.log_softmax(logits[:-1].float(), dim = -1)        # predict token t+1
    tgt = input_ids[0, 1:]                                     # [T-1]
    tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)    # [T-1]
    return tok_lp[n_prompt - 1:].sum()


def dpo_loss(lp_w_pol, lp_l_pol, lp_w_ref, lp_l_ref, beta):
    """
    Standard DPO loss: -log sigmoid(beta * [(pol_w-ref_w) - (pol_l-ref_l)]).

    Inputs:
        - lp_w_pol (Tensor): Policy logp of preferred completion
        - lp_l_pol (Tensor): Policy logp of dispreferred completion
        - lp_w_ref (Tensor): Reference logp of preferred completion
        - lp_l_ref (Tensor): Reference logp of dispreferred completion
        - beta (float): DPO temperature

    Outputs:
        - loss (Tensor): Scalar DPO loss
    """
    # Preference margin under the reference-normalized policy
    logits = beta * ((lp_w_pol - lp_w_ref) - (lp_l_pol - lp_l_ref))
    return -F.logsigmoid(logits)


@torch.no_grad()
def precompute_ref(model, tok, pairs, device):
    """
    Precompute reference (base, no LoRA) log-probs for every pair.

    Inputs:
        - model (nn.Module): Model with LoRA attached
        - tok (PreTrainedTokenizer): Tokenizer
        - pairs (list[dict]): Displacement pairs with instruction/yw/yl
        - device (str | torch.device): Device for forwards

    Outputs:
        - ref (list[tuple[float, float]]): (lp_w, lp_l) per pair
    """
    # Disable adapters so the reference is the frozen base
    set_lora_enabled(model, False)
    ref = []
    for pr in pairs:
        iw, nw = _tokenize_pair(tok, pr["instruction"], pr["yw"], device)
        il, nl = _tokenize_pair(tok, pr["instruction"], pr["yl"], device)
        ref.append((float(seq_logp(model, iw, nw)), float(seq_logp(model, il, nl))))
    set_lora_enabled(model, True)
    return ref


def train_dpo(model, tok, pairs, ref, lora_params, beta, lr, epochs, device, log_every = 50,
              opt8bit = False):
    """
    Train LoRA parameters with pairwise DPO on displacement pairs.

    Inputs:
        - model (nn.Module): Policy model with LoRA enabled
        - tok (PreTrainedTokenizer): Tokenizer
        - pairs (list[dict]): Displacement pairs
        - ref (list[tuple]): Precomputed reference log-probs
        - lora_params (list[Parameter]): Trainable LoRA parameters
        - beta (float): DPO temperature
        - lr (float): AdamW learning rate
        - epochs (int): Number of epochs
        - device (str | torch.device): Training device
        - log_every (int): Steps between loss logs
        - opt8bit (bool): Use bitsandbytes AdamW8bit when True

    Outputs:
        - None
    """
    # Prefer 8-bit AdamW at large rank to fit optimizer state
    if opt8bit:
        if bnb is None:
            raise ImportError("bitsandbytes is required when --opt8bit is set")
        opt = bnb.optim.AdamW8bit(lora_params, lr = lr)
        print("[opt] using bitsandbytes AdamW8bit")
    else:
        opt = torch.optim.AdamW(lora_params, lr = lr)
    set_lora_enabled(model, True)
    model.train()
    g = torch.Generator().manual_seed(0)
    step = 0
    for ep in range(epochs):
        perm = torch.randperm(len(pairs), generator = g).tolist()
        for i in perm:
            pr = pairs[i]
            iw, nw = _tokenize_pair(tok, pr["instruction"], pr["yw"], device)
            il, nl = _tokenize_pair(tok, pr["instruction"], pr["yl"], device)
            lp_w = seq_logp(model, iw, nw)
            lp_l = seq_logp(model, il, nl)
            loss = dpo_loss(lp_w, lp_l, ref[i][0], ref[i][1], beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % log_every == 0:
                print(f"    [dpo] ep{ep} step{step} loss={float(loss):.4f}")
    model.eval()


def fold_all(model, wrapped_names, cfg, beam_size = 5):
    """
    Fold each LoRALinear delta into base codes, then unwrap adapters.

    Inputs:
        - model (nn.Module): Model with LoRA wrappers
        - wrapped_names (list[str]): Wrapped module names
        - cfg (dict): AQLM quantization config
        - beam_size (int): Beam width for code updates

    Outputs:
        - changed (dict[str, float]): Per-module codes-changed fraction
    """
    # Fold then unwrap so the model is pure quantized with no FP adapter
    mods = dict(model.named_modules())
    changed = {}
    for name in wrapped_names:
        m = mods[name]
        if type(m).__name__ != "LoRALinear":
            continue
        delta = m.delta_weight().detach()                    # scaling * B@A  [out,in]
        frac = fold_and_writeback_(m.base, delta, cfg, beam_size = beam_size)
        changed[name] = frac
    unwrap_lora(model, wrapped_names)
    return changed


def eval_model(model, tok, n_eval, n_arc, label):
    """
    Evaluate ASR, ARC, and XSTest false-refusal for one model state.

    Inputs:
        - model (nn.Module): Model under evaluation
        - tok (PreTrainedTokenizer): Tokenizer
        - n_eval (int): WildJailbreak eval count
        - n_arc (int): ARC item count
        - label (str): Progress-bar prefix

    Outputs:
        - metrics (dict): asr, arc, xstest_fr
    """
    # ASR on WildJailbreak, then ARC and XSTest
    wj = load_wildjailbreak_prompts(n_samples = n_eval, config = "eval")
    beh = [_extract_behavior(it) for it in wj]
    resp = generate_only(model, tok, wj, desc = f"{label}/ASR")
    flags = classify_batch(beh, resp, device = "cuda")
    asr = 100.0 * sum(flags) / len(wj)
    arc = measure_arc(model, tok, load_arc_challenge(n_samples = n_arc))
    xs = eval_xstest(model, tok, load_xstest())
    print(f"  [{label}] ASR={asr:.1f}% ARC={arc:.0f}% XSTest-FR={xs['false_refusal_rate']:.1f}%")
    return {"asr": asr, "arc": arc, "xstest_fr": xs["false_refusal_rate"]}


def main():
    # Parse CLI arguments (defaults match official Q-Resafe llama7b recipe)
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", required = True, choices = ["read", "write", "full"])
    ap.add_argument("--model", default = "llama7b")
    ap.add_argument("--quantizer", default = "aqlm2")
    ap.add_argument("--r", type = int, default = 128)
    ap.add_argument("--alpha", type = float, default = 128.0, help = "LoRA alpha (official 128; our earlier runs used 256 = 2x scaling)")
    ap.add_argument("--beta", type = float, default = 0.01)
    ap.add_argument("--lr", type = float, default = 5e-7) #help = "DPO learning rate (official 5e-7; our earlier runs used 5e-6 = 10x)")
    ap.add_argument("--epochs", type = int, default = 1) #help = "DPO epochs (official 1; our earlier runs used 3)")
    ap.add_argument("--n_eval", type = int, default = 400)
    ap.add_argument("--n_arc", type = int, default = 100)
    ap.add_argument("--pairs_file", default = None, help = "displacement pairs JSON (default: qresafe_pairs_<model>_<quant>_wj.json)")
    ap.add_argument("--output", default = None)
    ap.add_argument("--tag", default = None,
                    help = "suffix appended to the output variant so hyperparameter-sweep "
                           "points do not overwrite each other, e.g. --tag lr5e-6 writes "
                           "outputs/trained_repair/qresafe_dpo_<model>_<quant>_<mask>_lr5e-6.json")
    ap.add_argument("--opt8bit", action = "store_true", default = False, help = "use 8-bit AdamW (bitsandbytes) to fit large rank (e.g. r=2048)")
    ap.add_argument("--save_adapter", action = "store_true", default = False, help = "save the large LoRA adapter .pt (off by default; only for re-folding)")
    ap.add_argument("--seed", type = int, default = DEFAULT_SEED) #help = "global RNG seed for reproducible LoRA init + DPO training")
    args = ap.parse_args()

    # Seed every RNG before LoRA init and DPO
    set_global_seed(args.seed)
    print(f"[seed] global seed = {args.seed}")

    # Resolve displacement pairs path
    device = "cuda"
    _SCRIPT = Path(__file__)
    pairs_cfg = RunConfig(model = args.model, quantizer = args.quantizer, variant = "wj")
    pairs_path = (
        Path(args.pairs_file)
        if args.pairs_file
        else find_result(
            Path("experiments/trained_repair/qresafe_pairs.py"), pairs_cfg
)
        or result_path(Path("experiments/trained_repair/qresafe_pairs.py"), pairs_cfg)
    )
    pairs = json.loads(pairs_path.read_text())["pairs"]
    print(f"[data] {len(pairs)} displacement pairs from {pairs_path} | mask={args.mask}")

    # Attach masked LoRA and train
    model, tok = load_quantized(args.model, quantizer = args.quantizer)
    cfg = quant_cfg(model)
    lora_params, wrapped = attach_masked_lora(model, args.mask, r = args.r, alpha = args.alpha)
    n_train = sum(p.numel() for p in lora_params)
    print(f"[lora] mask={args.mask} wrapped={len(wrapped)} trainable={n_train}")

    print("[dpo] precomputing reference log-probs...")
    ref = precompute_ref(model, tok, pairs, device)
    print(f"[dpo] training beta={args.beta} lr={args.lr} epochs={args.epochs}")
    train_dpo(model, tok, pairs, ref, lora_params, args.beta, args.lr, args.epochs, device,
              opt8bit = args.opt8bit)

    # Branch (1): evaluate FP+LoRA before the lossy fold
    set_lora_enabled(model, True)
    fp_lora = eval_model(model, tok, args.n_eval, args.n_arc, f"{args.mask}/FP+LoRA")

    # Snapshot adapter tensors and relative delta norms
    adapter = {}
    for n, m in model.named_modules():
        if type(m).__name__ == "LoRALinear":
            dw = m.delta_weight().detach()
            adapter[n] = {
                "A": m.lora_A.detach().cpu(),
                "B": m.lora_B.detach().cpu(),
                "scaling": m.scaling,
                "delta_relnorm": float(
                    dw.norm() / (dequantize_module(m.base, cfg["nbits"]).norm() + 1e-9)
                ),
            }
    if args.save_adapter:
        try:
            out_json = args.output or str(
                result_path(
                    _SCRIPT,
                    RunConfig(model = args.model, quantizer = args.quantizer, variant = args.mask),
                )
            )
            torch.save(adapter, out_json.replace(".json", "_adapter.pt"))
        except Exception as e:
            print(f"[warn] adapter save failed ({type(e).__name__}: {e}); continuing")
    rel = [v["delta_relnorm"] for v in adapter.values()]
    print(f"[delta] trained LoRA delta rel-norm: mean={sum(rel) / len(rel):.4f} "
          f"min={min(rel):.4f} max={max(rel):.4f}  (vs ~0.016 snap threshold seen in smoke)")

    # Fold into 2-bit codes and unwrap
    changed = fold_all(model, wrapped, cfg)
    mean_changed = sum(changed.values()) / max(len(changed), 1)
    print(f"[fold] mean codes-changed fraction = {mean_changed:.4f} "
          f"({'delta SURVIVES' if mean_changed > 1e-4 else 'delta SNAPPED AWAY'})")

    folded = eval_model(model, tok, args.n_eval, args.n_arc, f"{args.mask}/folded-2bit")

    # Persist result JSON
    run_cfg = RunConfig(
        model = args.model,
        quantizer = args.quantizer,
        variant = (f"{args.mask}_{args.tag}" if args.tag else args.mask),
        extra = {
            "r": args.r,
            "alpha": args.alpha,
            "beta": args.beta,
            "seed": args.seed,
            "lr": args.lr,
            "epochs": args.epochs,
            "n_eval": args.n_eval,
            "pairs_file": str(pairs_path),
        },
    )
    data = {
        "n_pairs": len(pairs),
        "wrapped_modules": len(wrapped),
        "mean_codes_changed": mean_changed,
        "fp_lora": fp_lora,
        "folded": folded,
        "baselines": {"fp32_asr": 45.5, "aqlm2_asr": 63.2, "fp32_xstest": 6.0},
    }
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents = True, exist_ok = True)
        out_path.write_text(json.dumps({**data, "manifest": run_cfg.to_dict()}, indent = 2))
    else:
        out_path = save_result(_SCRIPT, run_cfg, data)
    print(f"\n[saved] {out_path}")
    print(f"  branch signal: FP+LoRA ASR={fp_lora['asr']:.1f}  folded ASR={folded['asr']:.1f}  "
          f"codes-changed={mean_changed:.4f}  XSTest folded={folded['xstest_fr']:.1f}% (FP 6.0%)")


def _self_test():
    """
    CPU DPO-math gate without loading a model.

    Inputs:
        - None

    Outputs:
        - ok (bool): True if loss is monotone and neutral equals ln 2
    """
    # Prefer-yw beyond ref -> small loss; prefer-yl -> large loss
    ok = True
    good = float(dpo_loss(torch.tensor(0.0), torch.tensor(-2.0),
                          torch.tensor(0.0), torch.tensor(0.0), beta = 1.0))
    bad = float(dpo_loss(torch.tensor(-2.0), torch.tensor(0.0),
                         torch.tensor(0.0), torch.tensor(0.0), beta = 1.0))
    neutral = float(dpo_loss(torch.tensor(0.0), torch.tensor(0.0),
                             torch.tensor(0.0), torch.tensor(0.0), beta = 1.0))
    print(f"  dpo_loss good={good:.4f} neutral={neutral:.4f} bad={bad:.4f}")
    if not (good < neutral < bad):
        print("  FAIL: dpo_loss not monotone in the preference margin")
        ok = False
    if abs(neutral - 0.6931) > 1e-3:  # -log sigmoid(0) = ln 2
        print("  FAIL: neutral loss != ln2")
        ok = False
    print("\n  DPO MATH GATE:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.exit(0 if _self_test() else 1)
    main()
