# Forward hooks that ablate a direction from residual-stream layer outputs.

from core.residuals import get_inner


def make_ablation_hook(r_hat_cpu):
    """
    Build a forward hook that projects r_hat out of the residual stream.

    Inputs:
        - r_hat_cpu (torch.Tensor): Pre-normalized direction on CPU

    Outputs:
        - hook (callable): Forward hook applying h <- h - (h @ r) * r
    """
    # Closure over the CPU direction; move to activation device/dtype on the fly
    def hook(module, inputs, output):
        """
        Ablate r_hat from a layer's residual-stream output.

        Inputs:
            - module (nn.Module): Hooked module (unused)
            - inputs (tuple): Forward inputs (unused)
            - output (Tensor | tuple): Layer output or (hidden, ...)

        Outputs:
            - output (Tensor | tuple): Ablated output matching the input layout
        """
        # Preserve tuple vs tensor return layout
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output

        # Only ablate batched sequence residuals (B, T, D)
        if h.dim() != 3:
            return output
        r = r_hat_cpu.to(h.device).to(h.dtype)
        proj = (h @ r).unsqueeze(-1)
        h_new = h - proj * r
        if is_tuple:
            return (h_new,) + output[1:]
        return h_new

    return hook


def install_ablation(model, r_hat_cpu, layers):
    """
    Register ablation hooks on the listed residual-stream layers.

    Inputs:
        - model (torch.nn.Module): Model (possibly wrapped)
        - r_hat_cpu (torch.Tensor): Pre-normalized direction on CPU
        - layers (list): Layer indices to ablate

    Outputs:
        - handles (list): Forward-hook handles for later removal
    """
    # Resolve inner transformer that owns .layers
    inner = get_inner(model)
    handles = []

    # One hook per requested layer
    for L in layers:
        handles.append(
            inner.layers[L].register_forward_hook(make_ablation_hook(r_hat_cpu))
        )
    return handles


def remove_handles(handles):
    """
    Remove a list of forward-hook handles.

    Inputs:
        - handles (list): Hook handles from install_ablation

    Outputs:
        - None
    """
    # Detach every registered hook
    for h in handles:
        h.remove()
