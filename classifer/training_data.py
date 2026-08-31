"""
Shared data plumbing for the classifiers: batching and train/test splitting.

Extracted so ClassiferSmall and MultiHeadClassifier cannot drift apart on the
two decisions that most affect their numbers — how samples are batched off disk,
and how the train/test boundary is drawn.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


class ImageBatches(tf.keras.utils.PyDataset):
    """Batch generator over a memory-mapped image array.

    resized_expressions.npy is 9.6 GB for the RNAseqDB corpus. Loading it and
    then calling train_test_split copies it, peaking near 19 GB — enough to
    thrash a 16 GB machine. Indexing a memmap per batch keeps resident memory at
    roughly batch size, and costs nothing on a large VM.

    `labels` may be a dict of arrays (multi-head) or a single array (single-head);
    batches come back in the matching shape.
    """

    def __init__(self, images, labels, indices, batch_size, shuffle=False, seed=0, **kw):
        super().__init__(**kw)
        self.images = images
        self.labels = labels
        self.indices = np.asarray(indices)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        if shuffle:
            self.rng.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, i):
        # sorted so the memmap is read close to sequentially
        idx = np.sort(self.indices[i * self.batch_size:(i + 1) * self.batch_size])
        x = np.asarray(self.images[idx], dtype=np.float32)
        if isinstance(self.labels, dict):
            y = {k: v[idx] for k, v in self.labels.items()}
        else:
            y = self.labels[idx]
        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)


def donor_of(sample_id):
    """Donor/patient key. GTEX-1117F-0226-... -> GTEX-1117F;
    TCGA-GU-A42P-01A-... -> TCGA-GU-A42P."""
    parts = sample_id.split("-")
    return "-".join(parts[:2]) if sample_id.startswith("GTEX") else "-".join(parts[:3])


def make_split(strat_labels, frame, mode="stratified", test_size=0.25, seed=1):
    """Train/test indices under one of three regimes.

    stratified (default)
        Stratified on tissue x condition when both are available, else on the
        single label given. Some cells are genuinely thin — cervix has 13
        normals — so an unstratified split can leave a cell absent from one side
        entirely, making its per-class accuracy undefined rather than noisy.

    vinas
        Reproduces Viñas et al.'s procedure for this corpus exactly:
        np.random.seed(0), shuffle the sample axis, take a positional 75/25
        slice. No stratification and no donor grouping — their
        example_synthetic_data notebook calls the plain split_train_test(), not
        the patient-leak-aware split_train_test_v2() that also ships in their
        utils.

        The PROCEDURE matches; the PARTITION cannot. Their shuffle is seeded but
        permutes an array whose pre-shuffle order comes from os.listdir(), which
        is arbitrary and machine-dependent. Use this for like-for-like
        comparison, not to reproduce their exact rows.

    donor
        Groups by donor so no individual appears on both sides. This is the
        methodologically honest split and it is NOT what the reference paper did:
        98% of GTEx samples here share a donor with another sample (488 donors
        across 2,322 samples), and a random 75/25 split puts ~550 donors on both
        sides. Report it alongside `vinas`, not instead of it.

    Args:
        strat_labels: dict of one-hot arrays used to build the stratification
            key, or None to skip stratification.
        frame: label DataFrame; needs a sample_id column for mode="donor".
    """
    n = len(frame)
    idx = np.arange(n)

    if mode == "vinas":
        rng = np.random.RandomState(0)
        shuffled = idx.copy()
        rng.shuffle(shuffled)
        cut = int((1.0 - test_size) * n)
        return shuffled[:cut], shuffled[cut:]

    if mode == "donor":
        from sklearn.model_selection import GroupShuffleSplit
        groups = frame["sample_id"].map(donor_of).to_numpy()
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr, te = next(gss.split(idx, groups=groups))
        return idx[tr], idx[te]

    strat = None
    if strat_labels:
        if "tissue" in strat_labels and "condition" in strat_labels:
            key = (strat_labels["tissue"].argmax(1).astype(np.int64) * 100
                   + strat_labels["condition"].argmax(1))
        else:
            key = next(iter(strat_labels.values())).argmax(1)
        counts = np.bincount(key)
        if counts[counts > 0].min() >= 2:
            strat = key
    return train_test_split(idx, test_size=test_size, random_state=seed, stratify=strat)


def describe_split(frame, train_idx, val_idx, mode):
    """One-line summary including donor leakage, which the mode choice controls."""
    tr = set(frame.sample_id.iloc[train_idx].map(donor_of))
    te = set(frame.sample_id.iloc[val_idx].map(donor_of))
    shared = len(tr & te)
    note = ("  <- matches the reference paper, which did not group by donor"
            if mode != "donor" and shared else "")
    return (f"  split    : {len(train_idx)} train / {len(val_idx)} val  (mode={mode})\n"
            f"             {shared} donors appear on BOTH sides{note}")
