import random

import pandas as pd
from datasets import load_dataset


def load_harmbench_prompts(n_samples = None, cache_dir = None):
    """
    Load harmful behaviors from HarmBench (secondary safety benchmark) directly from GitHub.
    Dataset: https://github.com/centerforaisafety/HarmBench/blob/main/data/behavior_datasets/harmbench_behaviors_text_all.csv
    Reference: Mazeika et al., HarmBench (ICML 2024).
    Pinned revision: 8e1604d1171fe8a48d8febecd22f600e462bdcdd (recorded 2026-07-10)

    Returns list of str: harmful behavior strings for the HarmBench judge.
    """
    try:
        sha = "8e1604d1171fe8a48d8febecd22f600e462bdcdd"  # centerforaisafety/HarmBench, pinned for reproducibility
        url = f"https://raw.githubusercontent.com/centerforaisafety/HarmBench/{sha}/data/behavior_datasets/harmbench_behaviors_text_all.csv"
        df = pd.read_csv(url)
    except Exception as e:
        raise RuntimeError(f"[ERROR] Cannot load HarmBench CSV from GitHub: {e}") from e

    # The CSV usually has 'Behavior' column
    prompts = []
    for behavior in df["Behavior"].dropna():
        behavior = str(behavior).strip()
        if not behavior:
            continue
        prompts.append(behavior)
    if not prompts:
        raise RuntimeError(
            "[ERROR] HarmBench loaded but 'Behavior' field is empty, please check the CSV schema."
        )

    if n_samples:
        prompts = prompts[:n_samples]
    return prompts


# -- Safety: adversarial jailbreak prompts ------------------------------------
def load_wildjailbreak_prompts(n_samples = None, config = "train", cache_dir = None):
    """
    Load adversarial jailbreak prompts from WildJailbreak (NeurIPS 2024).

    Dataset: wildjailbreak (https://huggingface.co/datasets/allenai/wildjailbreak)
    Configs: "train" (262K examples, has vanilla+adversarial) or "eval" (2,210, no vanilla)

    Reference: Jiang et al., WildTeaming (NeurIPS 2024).
    Pinned revision: 5ddc12a7894f842b0619b8e1c7ee496b198af009 (recorded 2026-07-10)

    Uses the "adversarial_harmful" subset, prompts with jailbreak tactics composed in.

    Returns list of dicts containing:
      - prompt (str): the adversarial (jailbroken) prompt to send to the model
      - vanilla (str): the original harmful intent (short, used as behavior label for judge)
      - data_type (str): the type of the data
    """
    try:
        dataset = load_dataset(
            "allenai/wildjailbreak",
            config,
            revision="5ddc12a7894f842b0619b8e1c7ee496b198af009",  # pinned for reproducibility
            delimiter="\t",
            keep_default_na=False,
            cache_dir=cache_dir,
        )
        # Config defines one split; access via "train" key
        dataset = dataset["train"]
    except Exception as e:
        raise RuntimeError(
            f"[ERROR] Cannot load WildJailbreak (allenai/wildjailbreak, config='{config}'): {e}"
        ) from e

    # Filter for adversarial harmful prompts only
    items = []
    for row in dataset:
        dtype = row.get("data_type", "")
        if "adversarial_harmful" not in dtype:
            continue
        adv = row.get("adversarial", "").strip()
        if not adv:
            continue
        vanilla = row.get("vanilla", "").strip()
        items.append(
            {
                "prompt": adv,
                "vanilla": vanilla,
                "data_type": dtype,
            }
        )

    if not items:
        raise RuntimeError(
            "[ERROR] WildJailbreak loaded but no adversarial_harmful items found."
        )

    # Fixed seed ensures same prompts across phases in a single pipeline run
    random.Random(42).shuffle(items)
    if n_samples:
        items = items[:n_samples]
    return items


# -- Safety: harmless prompts (for refusal direction) -------------------------


def load_harmless_prompts(n_samples = None, cache_dir = None):
    """
    Load harmless prompts from Alpaca-cleaned.

    Dataset: gururise/AlpacaDataCleaned (origin repo of the cleaned Alpaca data; the HF
             'yahma/alpaca-cleaned' dataset is a mirror of this). Loaded as raw JSON from
             GitHub. Original Alpaca: Taori et al., Stanford Alpaca (2023).
    Pinned revision: d03c782bffd50ceeb1f4ef3c020129229ec4698c (recorded 2026-07-10)

    Returns list of str: instruction [+ newline + input if present].
    """
    try:
        sha = "d03c782bffd50ceeb1f4ef3c020129229ec4698c"  # gururise/AlpacaDataCleaned, pinned for reproducibility
        url = f"https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/{sha}/alpaca_data_cleaned.json"
        df = pd.read_json(url)
    except Exception as e:
        raise RuntimeError(
            f"[ERROR] Cannot load Alpaca-cleaned JSON from GitHub (gururise/AlpacaDataCleaned): {e}"
        ) from e

    prompts = []
    for _, item in df.iterrows():
        instr = str(item.get("instruction", "")).strip()
        inp = str(item.get("input", "")).strip()
        if not instr:
            continue
        if inp:
            prompts.append(f"{instr}\n{inp}")
        else:
            prompts.append(instr)

    if not prompts:
        raise RuntimeError(
            "[ERROR] Alpaca-cleaned loaded but no instruction fields found, please check the schema."
        )

    random.Random(42).shuffle(prompts)
    if n_samples:
        prompts = prompts[:n_samples]
    return prompts


def load_arc_challenge(n_samples = None, split="test", cache_dir = None):
    """
    ARC-Challenge: 4-way science multiple-choice task.

    Dataset: ai2_arc (https://huggingface.co/datasets/allenai/ai2_arc/viewer/ARC-Challenge/train?p=11)
    Reference: Clark et al., ARC (2018).
    Pinned revision: 210d026faf9955653af8916fad021475a3f00453 (recorded 2026-07-10)
    Returns list of dicts containing:
    - question (str): the question to send to the model
    - choices (list of str): the choices to send to the model
    - answer_idx (int): the index of the correct choice
    """
    try:
        dataset = load_dataset(
            "allenai/ai2_arc",
            "ARC-Challenge",
            split=split,
            revision="210d026faf9955653af8916fad021475a3f00453",  # pinned for reproducibility
            cache_dir=cache_dir,
        )
    except Exception as e:
        raise RuntimeError(
            f"[ERROR] Cannot load ARC-Challenge (allenai/ai2_arc/ARC-Challenge, split='{split}'): {e}"
        ) from e

    items = []
    for item in dataset:
        labels = item["choices"]["label"]
        texts = item["choices"]["text"]
        answer_key = item["answerKey"]
        try:
            answer_idx = labels.index(answer_key)
        except ValueError:
            continue  # Skips rare malformed entries where the answer key is not found in the label list
        items.append(
            {"question": item["question"], "choices": texts, "answer_idx": answer_idx}
        )

    if not items:
        raise RuntimeError("[ERROR] ARC-Challenge loaded but returned 0 valid items.")

    if n_samples:
        items = items[:n_samples]
    return items


# -- XSTest: over-refusal benchmark (Rottger et al. 2024) ---------------------


def load_xstest(n_samples = None, cache_dir = None):
    """
    XSTest: 450 prompts (250 'safe' designed-to-trick benign + 200 'unsafe' genuinely
    harmful) used to measure exaggerated safety behavior (over-refusal).

    Dataset: paul-rottger/xstest (original author repo, loaded as raw CSV from GitHub)
    Reference: Röttger et al., XSTest (NAACL 2024).
    Pinned revision: d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d (recorded 2026-07-10)

    Returns list of dicts containing:
    - prompt (str): the prompt to send to the model
    - type (str): "safe" or "unsafe"
    - category (str): the category of the prompt
    """
    try:
        sha = "d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d"  # paul-rottger/xstest, pinned for reproducibility
        url = f"https://raw.githubusercontent.com/paul-rottger/xstest/{sha}/xstest_prompts.csv"
        df = pd.read_csv(url)
    except Exception as e:
        raise RuntimeError(
            f"[ERROR] Cannot load XSTest CSV from GitHub (paul-rottger/xstest): {e}"
        ) from e

    # CSV schema: id, prompt, type, label, focus, note
    # 'label' is an explicit 'safe'/'unsafe' column, so no contrast-prefix heuristic needed:
    #   safe   -> 250 designed-to-trick-but-benign prompts (homonyms, figurative_language, ...)
    #   unsafe -> 200 genuinely harmful contrast prompts (contrast_homonyms, ...)
    items = []
    for _, row in df.iterrows():
        prompt = str(row.get("prompt", "")).strip()
        if not prompt:
            continue
        label = str(row.get("label", "")).strip().lower()
        items.append(
            {
                "prompt": prompt,
                "type": "safe" if label == "safe" else "unsafe",
                "category": str(row.get("type") or row.get("focus") or "").strip(),
            }
        )
    if not items:
        raise RuntimeError("XSTest loaded but returned 0 items -- check CSV schema.")
    if n_samples:
        items = items[:n_samples]
    return items


# -- C4: generic web text for the perplexity-only calibration baseline --------


def load_c4_calibration(n_samples = 64, max_chars = 1000, cache_dir = None):
    """
    Random C4 (Colossal Clean Crawled Corpus) passages for use as perplexity-style (no harmful/harmless contrast) calibration data.
    Reference: Raffel et al., C4/T5 (JMLR 2020).

    Dataset: allenai/c4 (https://huggingface.co/datasets/allenai/c4/viewer/en/train?p=11)
    Pinned revision: 1588ec454efa1a09f29cd18ddd04fe05fc8653a2 (recorded 2026-07-10)

    Returns a list of strings, each <= max_chars characters, suitable as the 'user turn' in a chat template. The teacher will generate continuations.

    """
    try:
        ds = load_dataset(
            "allenai/c4",
            "en",
            split="train",
            revision="1588ec454efa1a09f29cd18ddd04fe05fc8653a2",  # pinned for reproducibility
            streaming=True,
            cache_dir=cache_dir,
        )
    except Exception as e:
        raise RuntimeError(f"[ERROR] Cannot load C4 (allenai/c4, 'en'): {e}") from e

    prompts = []
    for doc in ds:
        text = (doc.get("text") or "").strip()
        if len(text) < 100:
            continue  # skip very short documents
        prompts.append(text[:max_chars])
        if len(prompts) >= n_samples:
            break
    if not prompts:
        raise RuntimeError("[ERROR] C4 streamed but returned 0 usable documents.")
    return prompts
