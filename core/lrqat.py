# Quant-domain low-rank repair (LR-QAT) for SafeQuant v2.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.residuals import get_inner


# -- straight-through round ---------------------------------------------------
# Forward rounds to nearest integer; backward passes the gradient through
class _RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        """
        Round tensor to nearest integer (STE forward).

        Inputs:
            - ctx: Autograd context (unused)
            - x (torch.Tensor): Real-valued tensor

        Outputs:
            - y (torch.Tensor): Rounded tensor
        """
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Straight-through backward: pass gradient unchanged.

        Inputs:
            - ctx: Autograd context (unused)
            - grad_output (torch.Tensor): Upstream gradient

        Outputs:
            - grad_input (torch.Tensor): Same as grad_output
        """
        return grad_output


round_ste = _RoundSTE.apply


# -- symmetric group-wise integer PTQ -----------------------------------------
def symmetric_int_ptq(weight, n_bits, group_size = 128, eps = 1e-8):
    """
    Group-wise symmetric signed INT-b post-training quantization.

    Inputs:
        - weight (torch.Tensor): Dense weight to quantize
        - n_bits (int): Integer bitwidth
        - group_size (int): Group size along in_features (or full row if invalid)
        - eps (float): Floor for absmax scale

    Outputs:
        - W_int (torch.Tensor): Integer codes
        - scale (torch.Tensor): Per-group scales [out, n_groups]
        - qmin (int): Minimum representable integer
        - qmax (int): Maximum representable integer
        - group_size (int): Effective group size used
    """
    # Signed integer range for n_bits
    qmax = 2 ** (n_bits - 1) - 1
    qmin = -(2 ** (n_bits - 1))

    # Materialize float weight and shape
    w = weight.detach().float()
    out_f, in_f = w.shape

    # Fall back to whole-row groups when group_size is unset or too large
    if group_size is None or group_size <= 0 or group_size >= in_f:
        group_size = in_f
    if in_f % group_size != 0:
        raise ValueError(f"in_features={in_f} not divisible by group_size={group_size}")
    n_groups = in_f // group_size

    # Per-group absmax scales and rounded codes
    wg = w.view(out_f, n_groups, group_size)
    amax = wg.abs().amax(dim = 2, keepdim = True).clamp_min(eps)
    scale_g = amax / max(qmax, 1)
    W_int = torch.clamp(torch.round(wg / scale_g), qmin, qmax).view(out_f, in_f)
    scale = scale_g.squeeze(2).to(torch.float32)
    return W_int, scale, qmin, qmax, group_size


class LRQATLinear(nn.Module):
    def __init__(
        self,
        weight_fp,
        bias,
        n_bits,
        rank,
        alpha,
        device,
        compute_dtype = torch.float16,
        group_size = 128,
    ):
        """
        Build an INT-b grid linear plus trainable integer-domain low-rank adapter.

        Inputs:
            - weight_fp (torch.Tensor): Dense FP weight to quantize
            - bias (torch.Tensor | None): Optional bias
            - n_bits (int): Integer bitwidth
            - rank (int): LoRA rank
            - alpha (float): LoRA alpha (scaling = alpha / rank)
            - device (torch.device): Parameter device
            - compute_dtype (torch.dtype): Bias / compute dtype
            - group_size (int): Group size for symmetric PTQ

        Outputs:
            - None
        """
        super().__init__()

        # Shape bookkeeping
        out_f, in_f = weight_fp.shape
        self.in_features = in_f
        self.out_features = out_f
        self.n_bits = int(n_bits)
        self.rank = int(rank)
        self.lora_scaling = float(alpha) / float(rank)

        # Freeze a group-wise integer grid from the FP weight
        W_int, scale, qmin, qmax, gs = symmetric_int_ptq(
            weight_fp, n_bits, group_size = group_size
        )
        self.group_size = int(gs)
        self.n_groups = in_f // self.group_size
        self.register_buffer("W_int", W_int.to(device))
        self.register_buffer("scale", scale.to(device))
        self.qmin = int(qmin)
        self.qmax = int(qmax)

        # Trainable low-rank adapter in fp32
        self.lora_A = nn.Parameter(
            torch.zeros(out_f, self.rank, device = device, dtype = torch.float32)
        )
        self.lora_Bt = nn.Parameter(
            torch.empty(self.rank, in_f, device = device, dtype = torch.float32)
        )
        nn.init.kaiming_uniform_(self.lora_Bt, a = math.sqrt(5))

        # Optional bias buffer
        if bias is not None:
            self.register_buffer("bias", bias.detach().to(device).to(compute_dtype))
        else:
            self.bias = None

        self.merged = False

    def _delta(self):
        """
        Compute the scaled low-rank adapter delta in integer space.

        Inputs:
            - None

        Outputs:
            - delta (torch.Tensor): (lora_A @ lora_Bt) * scaling
        """
        return (self.lora_A @ self.lora_Bt) * self.lora_scaling

    def _scale_full(self):
        """
        Expand group-wise scale [out, n_groups] to [out, in].

        Inputs:
            - None

        Outputs:
            - scale (torch.Tensor): Broadcastable / expanded scale
        """
        if self.group_size >= self.in_features:
            return self.scale
        return self.scale.repeat_interleave(self.group_size, dim = 1)

    def _w_hat(self):
        """
        Differentiable repaired weight (un-merged path).

        Inputs:
            - None

        Outputs:
            - w_hat (torch.Tensor): scale * clamp(W_int + round_ste(delta))
        """
        W_int_hat = torch.clamp(
            self.W_int + round_ste(self._delta()), self.qmin, self.qmax
        )
        return self._scale_full() * W_int_hat

    def forward(self, x):
        """
        Apply the (merged or repaired) quantized linear to x.

        Inputs:
            - x (torch.Tensor): Input activations

        Outputs:
            - y (torch.Tensor): Linear output
        """
        # Merged path uses frozen codes only
        if self.merged:
            weight = (self._scale_full() * self.W_int).to(x.dtype)
        else:
            weight = self._w_hat().to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)

    @torch.no_grad()
    def merge_and_requantize(self):
        """
        Fold the trained adapter into the integer grid (LR-QAT merge).

        Inputs:
            - None

        Outputs:
            - None
        """
        # No-op if already merged
        if self.merged:
            return

        # Fuse rounded delta into W_int and clear the adapter
        C = self._delta()
        W_int_fused = torch.clamp(self.W_int + torch.round(C), self.qmin, self.qmax)
        self.W_int.copy_(W_int_fused)
        self.lora_A.data.zero_()
        self.merged = True

    @torch.no_grad()
    def closed_form_fill(self, weight_fp, rank = None):
        """
        Training-free repair: fold the top-r SVD of the quantization residual
        (W_fp - dequant(W_int)) into the INT grid. This is the closed-form
        analogue of the distilled low-rank adapter -- same slot, same bit-width,
        no gradient steps. The correction is requantized into the existing grid,
        so effective bits are unchanged.

        Inputs:
            - weight_fp (torch.Tensor): FP16/FP32 target (teacher) weight
            - rank (int | None): correction rank (defaults to the adapter rank)

        Outputs:
            - None
        """
        r = self.rank if rank is None else int(rank)
        scale_full = self._scale_full().float()
        W_deq = scale_full * self.W_int.float()
        R = weight_fp.float().to(W_deq.device) - W_deq
        U, S, Vh = torch.linalg.svd(R, full_matrices = False)
        r = min(r, S.numel())
        R_r = (U[:, :r] * S[:r]) @ Vh[:r, :]
        delta_int = torch.round(R_r / scale_full)
        self.W_int.copy_(torch.clamp(self.W_int + delta_int, self.qmin, self.qmax))
        self.lora_A.data.zero_()
        self.merged = True

    def effective_bits(self):
        """
        Effective storage bits/weight once fused (codes + fp16 scales).

        Inputs:
            - None

        Outputs:
            - effective_bits (float): Mean bits per weight element
        """
        numel = self.W_int.numel()
        scale_bits = 16 * self.scale.numel()
        return (self.n_bits * numel + scale_bits) / numel

    def trainable_parameters(self):
        """
        Return the trainable low-rank adapter parameters.

        Inputs:
            - None

        Outputs:
            - params (list): [lora_A, lora_Bt]
        """
        return [self.lora_A, self.lora_Bt]

    def extra_repr(self):
        """
        Compact string for Module printing.

        Inputs:
            - None

        Outputs:
            - extra_repr (str): Summary of shapes and hyperparams
        """
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"n_bits={self.n_bits}, group_size={self.group_size}, "
            f"rank={self.rank}, scaling={self.lora_scaling:.3g}, "
            f"merged={self.merged}"
        )


# -- module tree helpers (kept local so core/ has no diagnostics dependency) ---
def _get_submodule(layer, dotted):
    """
    Resolve a dotted submodule path under a layer.

    Inputs:
        - layer (torch.nn.Module): Parent module
        - dotted (str): Dotted attribute path

    Outputs:
        - m (torch.nn.Module): Resolved submodule
    """
    m = layer
    for p in dotted.split("."):
        m = getattr(m, p)
    return m


def _set_submodule(layer, dotted, new_module):
    """
    Replace a dotted submodule path under a layer.

    Inputs:
        - layer (torch.nn.Module): Parent module
        - dotted (str): Dotted attribute path
        - new_module (torch.nn.Module): Replacement module

    Outputs:
        - None
    """
    parts = dotted.split(".")
    parent = layer
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def inject_lrqat_adapters(
    model,
    proj_paths,
    n_bits,
    rank,
    alpha,
    get_weight_fn,
    compute_dtype = torch.float16,
    group_size = 128,
    layer_range = None,
    verbose = True,
):
    """
    Replace each selected QuantLinear with an LRQATLinear (Route A1).

    Inputs:
        - model (torch.nn.Module): Model to modify in place
        - proj_paths (list): Projection dotted paths per layer
        - n_bits (int): Integer bitwidth
        - rank (int): LoRA rank
        - alpha (float): LoRA alpha
        - get_weight_fn (callable): Weight extractor (module, as_dtype=...)
        - compute_dtype (torch.dtype): Bias / compute dtype
        - group_size (int): PTQ group size
        - layer_range (tuple | None): (lo, hi) to inject only on layers[lo:hi]
          (depth-band hybrid); None injects on all layers
        - verbose (bool): Print injection summary

    Outputs:
        - injected (list): Injected LRQATLinear modules
        - trainable (list): Trainable adapter parameter tensors
    """
    # Freeze base and resolve layer stack / device
    inner = get_inner(model)
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    for p in model.parameters():
        p.requires_grad_(False)

    # Swap each selected projection for an LRQATLinear (optionally band-limited)
    lo, hi = layer_range if layer_range else (0, n_layers)
    injected = []
    for l in range(n_layers):
        if not (lo <= l < hi):
            continue
        layer = inner.layers[l]
        for proj in proj_paths:
            mod = _get_submodule(layer, proj)
            w = get_weight_fn(mod, as_dtype = torch.float32).to(device)
            bias = getattr(mod, "bias", None)
            bias = bias if (bias is not None) else None
            new = LRQATLinear(
                w,
                bias,
                n_bits = n_bits,
                rank = rank,
                alpha = alpha,
                device = device,
                compute_dtype = compute_dtype,
                group_size = group_size,
            )
            new._lrqat_layer = l
            new._lrqat_proj = proj
            _set_submodule(layer, proj, new)
            injected.append(new)

    # Collect trainable adapter params
    trainable = []
    for m in injected:
        for p in m.trainable_parameters():
            p.requires_grad_(True)
            trainable.append(p)

    if verbose:
        n_train = sum(p.numel() for p in trainable)
        band = f"layers[{lo},{hi})" if layer_range else "all-layers"
        print(
            f"  Injected {len(injected)} LRQATLinear adapters "
            f"(n_bits={n_bits}, group_size={group_size}, rank={rank}, alpha={alpha}, band={band})"
        )
        print(
            f"  Trainable adapter params: {n_train:,} "
            f"({n_train / 1e6:.2f}M) across {len(proj_paths)} proj types"
        )
    return injected, trainable


def merge_all(injected, verbose = True):
    """
    Fuse + requantize every injected adapter into its INT-b grid.

    Inputs:
        - injected (list): Injected LRQATLinear modules
        - verbose (bool): Print merge summary

    Outputs:
        - None
    """
    for m in injected:
        m.merge_and_requantize()
    if verbose:
        print(
            f"  Fused {len(injected)} adapters back into INT-{injected[0].n_bits} grid"
        )


def closed_form_all(injected, get_teacher_weight, rank = None, verbose = True):
    """
    Fill every injected adapter with a closed-form low-rank residual correction.

    Inputs:
        - injected (list): Injected LRQATLinear modules
        - get_teacher_weight (callable): module -> FP target weight tensor
        - rank (int | None): correction rank (defaults to each adapter's rank)
        - verbose (bool): Print summary

    Outputs:
        - None
    """
    for m in injected:
        m.closed_form_fill(get_teacher_weight(m), rank = rank)
    if verbose:
        used_r = rank if rank is not None else injected[0].rank
        print(
            f"  Closed-form filled {len(injected)} adapters "
            f"(top-{used_r} SVD residual, fused into INT-{injected[0].n_bits} grid)"
        )


@torch.no_grad()
def activation_aware_fill(injected, student, teacher, calib_blocks, device,
                          ridge = 1e-2, verbose = True):
    """
    Training-free OUTPUT-matching repair (the fix for weight-SVD's failure).

    For each injected role projection, accumulate over calibration blocks the
    second moment of the student's quantized INPUT activations (A = sum x x^T)
    and their cross moment with the teacher's FP OUTPUT (B = sum y x^T), then set
    the projection to the ridge least-squares weight W* = B (A + lam I)^-1 -- the
    weight that best maps the quantized inputs to the FP outputs -- and requantize
    W* into the INT grid. Because W* is a COMPENSATORY target (matches outputs,
    not weights), it differs from W_fp by integer-level amounts and survives the
    grid rounding that zeroes out the sub-step weight-SVD correction.

    Inputs:
        - injected (list): Injected LRQATLinear modules (need ._lrqat_layer/_proj)
        - student (nn.Module): quantized student holding the injected modules
        - teacher (nn.Module): frozen FP teacher (same architecture)
        - calib_blocks (Tensor): [n_blocks, seq_len] CPU token blocks
        - device: forward device
        - ridge (float): ridge strength, relative to the mean diagonal of A
        - verbose (bool): print progress

    Outputs:
        - None
    """
    t_inner = get_inner(teacher)

    def _teacher_mod(m):
        sub = t_inner.layers[m._lrqat_layer]
        for part in m._lrqat_proj.split("."):
            sub = getattr(sub, part)
        return sub

    # Pair each teacher projection module with its injected student counterpart.
    tmod_to_inj = {_teacher_mod(m): m for m in injected}

    A = {id(m): None for m in injected}   # in x in  (student input second moment)
    B = {id(m): None for m in injected}   # out x in (teacher-output x student-input)
    xcache = {}                           # id(inj) -> this block's flattened input

    def s_pre_hook(mod, args):
        x = args[0]
        xcache[id(mod)] = x.detach().reshape(-1, x.shape[-1]).float()

    def t_out_hook(mod, args, out):
        inj = tmod_to_inj[mod]
        if id(inj) not in xcache:
            return
        y = out.detach().reshape(-1, out.shape[-1]).float()
        x = xcache[id(inj)]
        aa, bb = x.transpose(0, 1) @ x, y.transpose(0, 1) @ x
        A[id(inj)] = aa if A[id(inj)] is None else A[id(inj)] + aa
        B[id(inj)] = bb if B[id(inj)] is None else B[id(inj)] + bb

    s_handles = [m.register_forward_pre_hook(s_pre_hook) for m in injected]
    t_handles = [tm.register_forward_hook(t_out_hook) for tm in tmod_to_inj]

    student.eval()
    teacher.eval()
    nb = calib_blocks.size(0)
    for i in range(nb):
        xb = calib_blocks[i:i + 1].to(device)
        xcache.clear()
        student(input_ids = xb)   # fills xcache via student pre-hooks
        teacher(input_ids = xb)   # accumulates A / B via teacher output hooks
        if verbose and (i + 1) % 8 == 0:
            print(f"    calib block {i + 1}/{nb}")
    for h in s_handles + t_handles:
        h.remove()

    # Solve the ridge least-squares per projection and requantize into the grid.
    for m in injected:
        a, b = A[id(m)], B[id(m)]
        if a is None:
            continue
        lam = ridge * (a.diagonal().mean().item() + 1e-8)
        eye = torch.eye(a.shape[0], device = a.device, dtype = a.dtype)
        Wstar = torch.linalg.solve(a + lam * eye, b.transpose(0, 1)).transpose(0, 1)
        scale_full = m._scale_full().float()
        W_int_new = torch.clamp(torch.round(Wstar / scale_full), m.qmin, m.qmax)
        m.W_int.copy_(W_int_new.to(m.W_int.dtype))
        m.lora_A.data.zero_()
        m.merged = True
        A[id(m)] = B[id(m)] = None
        del a, b, eye, Wstar, W_int_new
        torch.cuda.empty_cache()
    if verbose:
        print(f"  Activation-aware filled {len(injected)} projections "
              f"(ridge LS output-matching, requantized into INT-{injected[0].n_bits} grid)")


def injected_effective_bits(injected):
    """
    Mean effective bits/weight across all injected (repaired) matrices.

    Inputs:
        - injected (list): Injected LRQATLinear modules

    Outputs:
        - effective_bits (float): Mean bits/weight, or nan if empty
    """
    if not injected:
        return float("nan")
    return sum(m.effective_bits() for m in injected) / len(injected)
