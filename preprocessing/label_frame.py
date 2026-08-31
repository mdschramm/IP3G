"""
Multi-attribute label encoding.

The GTEx pipeline carries one categorical label per sample (body site, or
primary site) and encodes it with preprocess_data.get_y_train. The combined
TCGA+GTEx corpus carries several at once — tissue, condition, subtype, source,
and later sex and age — so a single label array no longer describes a sample.

This module is the multi-attribute equivalent: it turns a label frame
(one row per sample, one column per attribute) into one one-hot array per
attribute plus a vocabulary file recording the index -> value mapping.

WHY THE VOCABULARY IS PERSISTED
    get_y_train derives class indices from sorted(set(phenotypes)) at encode
    time and nothing writes them down, so recovering "what is class 40?" means
    re-deriving the sort from the original phenotype array — and an upstream
    change silently renumbers every class. (That exact failure cost a 96% ->
    2% accuracy drop before the sort was made explicit.) Here the ordering is
    computed once, written to attribute_vocab.json, and read back by every
    consumer instead of being recomputed.

NULL TOKENS ARE NOT PART OF THE VOCABULARY
    Classifier-free guidance needs an "any value" token per attribute. That
    token lives at index len(vocab[attr]) — one past the end — and is added by
    the diffusion model's embedding tables, not here. Keep it distinct from
    real values that happen to mean absence: subtype "none" means "this is a
    normal sample, it has no cancer subtype", which is a fact about the sample.
    The null token means "don't condition on subtype at all".

USAGE:
    from preprocessing.label_frame import build_vocab, one_hot, save_vocab

    vocab = build_vocab(labels, ATTRIBUTES)
    y_tissue = one_hot(labels, "tissue", vocab)
    save_vocab(vocab, config.attribute_vocab_path)
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd


def build_vocab(labels: pd.DataFrame, attributes) -> dict[str, list[str]]:
    """Ordered value list per attribute.

    Values are sorted so the encoding is deterministic across runs and
    machines — the same reason get_y_train sorts. Order is then frozen by
    persisting it; never re-derive it downstream.
    """
    vocab = {}
    for attr in attributes:
        if attr not in labels.columns:
            raise KeyError(f"Attribute {attr!r} not in label frame: {list(labels.columns)}")
        values = labels[attr]
        if values.isna().any():
            raise ValueError(f"Attribute {attr!r} has missing values")
        vocab[attr] = sorted(values.unique().tolist())
    return vocab


def one_hot(labels: pd.DataFrame, attribute: str, vocab: dict[str, list[str]]) -> np.ndarray:
    """One-hot encode one attribute against a fixed vocabulary.

    Returns:
        (N_samples, len(vocab[attribute])) float32
    """
    values = vocab[attribute]
    index = {v: i for i, v in enumerate(values)}
    col = labels[attribute]

    unknown = set(col.unique()) - set(values)
    if unknown:
        raise ValueError(f"Values absent from {attribute!r} vocabulary: {sorted(unknown)}")

    out = np.zeros((len(col), len(values)), dtype=np.float32)
    out[np.arange(len(col)), col.map(index).to_numpy()] = 1.0
    return out


def integer_codes(labels: pd.DataFrame, attributes, vocab: dict[str, list[str]]) -> np.ndarray:
    """Attributes as an integer matrix, the form the diffusion model consumes.

    Returns:
        (N_samples, len(attributes)) int32, column order matching `attributes`.
        The CFG null token for column i is len(vocab[attributes[i]]).
    """
    cols = []
    for attr in attributes:
        index = {v: i for i, v in enumerate(vocab[attr])}
        cols.append(labels[attr].map(index).to_numpy(dtype=np.int32))
    return np.stack(cols, axis=1)


def attribute_sizes(vocab: dict[str, list[str]], attributes=None) -> dict[str, int]:
    """Number of real (non-null) values per attribute."""
    keys = attributes if attributes is not None else vocab.keys()
    return {attr: len(vocab[attr]) for attr in keys}


def save_vocab(vocab: dict[str, list[str]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(vocab, fh, indent=2)


def load_vocab(path: str) -> dict[str, list[str]]:
    with open(path) as fh:
        return json.load(fh)


def describe(labels: pd.DataFrame, attributes, vocab: dict[str, list[str]]) -> str:
    """Human-readable per-attribute value counts, for logs and gate checks."""
    lines = []
    for attr in attributes:
        counts = labels[attr].value_counts()
        lines.append(f"  {attr} ({len(vocab[attr])} values)")
        for value in vocab[attr]:
            lines.append(f"      {value:<18} {counts.get(value, 0):>6}")
    return "\n".join(lines)
