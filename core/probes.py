# Linear/MLP refusal probes on residual streams.

from __future__ import annotations

# Front Matter
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

try:
    from core.seeds import DEFAULT_SEED as SEED
except ModuleNotFoundError:
    SEED = 42


def make_probe(probe_type):
    """
    Construct an unfitted sklearn probe of the requested type.

    Inputs:
        - probe_type (str): "linear" or "mlp"

    Outputs:
        - probe (LogisticRegression | MLPClassifier): Unfitted estimator
    """
    # Linear logistic regression probe
    if probe_type == "linear":
        return LogisticRegression(
            C = 1.0,
            max_iter = 2000,
            n_jobs = 1,
            random_state = SEED,
        )

    # Small MLP probe with early stopping
    if probe_type == "mlp":
        return MLPClassifier(
            hidden_layer_sizes = (256,),
            activation = "relu",
            solver = "adam",
            alpha = 1e-3,
            batch_size = 64,
            learning_rate_init = 1e-3,
            max_iter = 200,
            early_stopping = True,
            validation_fraction = 0.1,
            random_state = SEED,
        )
    raise ValueError(f"Unknown probe_type: {probe_type}")


def fit_probe_per_layer(
    harmful_by_layer, harmless_by_layer, n_folds = 5, probe_type = "linear"
):
    """
    Stratified k-fold probe accuracy per residual layer.

    Inputs:
        - harmful_by_layer (dict): Layer -> harmful residual tensors
        - harmless_by_layer (dict): Layer -> harmless residual tensors
        - n_folds (int): Number of CV folds
        - probe_type (str): "linear" or "mlp"

    Outputs:
        - results (dict): Layer -> mean/std and class-wise accuracies
    """
    # Stable layer order and output dict
    layers = sorted(harmful_by_layer.keys())
    results = {}

    for layer_idx in layers:
        # Stack residuals and build binary labels (1 = harmful)
        h = np.stack([t.cpu().numpy() for t in harmful_by_layer[layer_idx]])
        s = np.stack([t.cpu().numpy() for t in harmless_by_layer[layer_idx]])
        X = np.concatenate([h, s], axis = 0).astype(np.float64)
        y = np.concatenate([np.ones(len(h)), np.zeros(len(s))], axis = 0)
        skf = StratifiedKFold(n_splits = n_folds, shuffle = True, random_state = SEED)
        fold_accs, fold_h, fold_s = [], [], []

        # One probe fit per fold
        for tr_idx, te_idx in skf.split(X, y):
            Xtr, ytr = X[tr_idx], y[tr_idx]
            Xte, yte = X[te_idx], y[te_idx]

            # MLP needs feature scaling
            if probe_type == "mlp":
                scaler = StandardScaler()
                Xtr = scaler.fit_transform(Xtr)
                Xte = scaler.transform(Xte)
            clf = make_probe(probe_type)
            clf.fit(Xtr, ytr)
            fold_accs.append(clf.score(Xte, yte))
            h_mask = yte == 1
            s_mask = yte == 0

            # Class-conditional accuracies when the fold has that class
            if h_mask.any():
                fold_h.append(clf.score(Xte[h_mask], yte[h_mask]))
            if s_mask.any():
                fold_s.append(clf.score(Xte[s_mask], yte[s_mask]))

        # Aggregate fold metrics for this layer
        results[layer_idx] = {
            "mean_acc": float(np.mean(fold_accs)),
            "std_acc": float(np.std(fold_accs)),
            "harmful_acc": float(np.mean(fold_h)) if fold_h else float("nan"),
            "harmless_acc": float(np.mean(fold_s)) if fold_s else float("nan"),
        }
    return results


def fit_probe_get_weights(harmful_by_layer, harmless_by_layer):
    """
    Fit one logistic regression per layer on all data and return weight + bias.

    Inputs:
        - harmful_by_layer (dict): Layer -> harmful residual tensors
        - harmless_by_layer (dict): Layer -> harmless residual tensors

    Outputs:
        - out (dict): Layer -> weight, bias, train_acc
    """
    # Stable layer order
    layers = sorted(harmful_by_layer.keys())
    out = {}

    # Full-data linear probe per layer
    for layer_idx in layers:
        h = np.stack([t.cpu().numpy() for t in harmful_by_layer[layer_idx]])
        s = np.stack([t.cpu().numpy() for t in harmless_by_layer[layer_idx]])
        X = np.concatenate([h, s], axis = 0).astype(np.float64)
        y = np.concatenate([np.ones(len(h)), np.zeros(len(s))], axis = 0)

        clf = LogisticRegression(C = 1.0, max_iter = 2000, random_state = SEED)
        clf.fit(X, y)
        w = clf.coef_[0]

        out[layer_idx] = {
            "weight": w.tolist(),
            "bias": float(clf.intercept_[0]),
            "train_acc": float(clf.score(X, y)),
        }
    return out


def per_layer_probe_accuracy(residuals_by_layer, labels):
    """
    5-fold CV accuracy per layer for a logistic-regression probe.

    Inputs:
        - residuals_by_layer (dict): Layer -> residual tensors
        - labels (list): Per-prompt binary labels

    Outputs:
        - accs (dict): Layer -> mean fold accuracy (nan if single class)
    """
    # Shared fold splitter and label array
    layers = sorted(residuals_by_layer.keys())
    accs = {}
    skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = SEED)
    y = np.asarray(labels, dtype = np.int64)

    for layer_idx in layers:
        X = np.stack([t.numpy() for t in residuals_by_layer[layer_idx]]).astype(
            np.float64
        )

        # Stratified CV needs both classes
        if len(np.unique(y)) < 2:
            accs[layer_idx] = float("nan")
            continue

        fold_accs = []
        for train_idx, test_idx in skf.split(X, y):
            clf = LogisticRegression(
                C = 1.0, max_iter = 2000, n_jobs = 1, random_state = SEED
            )
            clf.fit(X[train_idx], y[train_idx])
            fold_accs.append(clf.score(X[test_idx], y[test_idx]))
        accs[layer_idx] = float(np.mean(fold_accs))
    return accs
