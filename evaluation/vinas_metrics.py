"""
Adapter over Viñas et al.'s evaluation metrics (external/adversarial-gene-expression).

Their code is used for the *logic* — the γ coefficient, hierarchical clustering
and cophenetic comparison are theirs, unmodified, so our numbers are computed the
same way as their published 0.920 / 0.215. This module only:

  1. puts the pinned submodule on sys.path,
  2. swaps two primitives that do not scale to 18,154 genes (both verified
     bit-identical — see test_vinas_equivalence.py),
  3. adds the two things §5.2 describes but the repo does not contain.

THE TWO SUBSTITUTIONS
    upper_diag_list       Their version builds ~3.1x the input matrix in
                          temporaries (np.triu + a full NaN matrix + np.tril +
                          add + ravel + boolean mask). At n=18,154 the input is
                          2.64 GB and the peak is ~10.5 GB. m[np.triu_indices(n, 1)]
                          is the same values at ~1.5x.

    dendrogram_distance   Their version is a pure-Python O(n^2) double loop over
                          cluster member pairs plus a dense n x n matrix. It
                          computes exactly the cophenetic distances, which
                          scipy.cluster.hierarchy.cophenet returns directly from
                          the same linkage matrix, in C, condensed.

    Both are installed by rebinding the module attribute, so their own
    correlations_list / compare_cophenetic pick them up without those functions
    being touched.

WHAT THE REPO DOES NOT HAVE
    sampling_bound()  The 0.947 / 0.222 "bound" rows. Present only as notebook
                      cells (gtex_tcga_analysis_2, cells 543-553), not in utils.py.
    tstr_scores()     The §5.2.2 train-on-synthetic-test-on-real MLP. Absent
                      entirely — roc_auc_score and f1_score are imported by the
                      notebooks and never called.

LINKAGE: scipy, not fastcluster. Their notebook overrides utils.hierarchical_clustering
to use fastcluster.linkage; we keep scipy.linkage with the same method='complete'
rather than add a dependency that forces a base-image rebuild. Same algorithm.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.cluster.hierarchy import cophenet

_SUBMODULE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "external", "adversarial-gene-expression",
)


def _load_vinas_utils():
    if not os.path.isdir(_SUBMODULE):
        raise RuntimeError(
            f"Metric submodule missing at {_SUBMODULE}.\n"
            "Initialise it with:\n"
            "  git submodule update --init external/adversarial-gene-expression"
        )
    if _SUBMODULE not in sys.path:
        sys.path.insert(0, _SUBMODULE)
    import utils as vinas  # noqa: E402  (path must be set first)
    return vinas


def _fast_upper_diag_list(m_):
    """Values above the diagonal, row-major. Equivalent to theirs at ~1.5x memory."""
    m_ = np.asarray(m_)
    return m_[np.triu_indices(m_.shape[0], k=1)]


def _fast_dendrogram_distance(l_matrix, condensed=True):
    """Cophenetic distances from a linkage matrix. Equivalent to theirs, in C."""
    d = cophenet(l_matrix)
    if condensed:
        return d
    from scipy.spatial.distance import squareform
    return squareform(d)


_vinas = _load_vinas_utils()
_ORIGINAL = {
    "upper_diag_list": _vinas.upper_diag_list,
    "dendrogram_distance": _vinas.dendrogram_distance,
}
_vinas.upper_diag_list = _fast_upper_diag_list
_vinas.dendrogram_distance = _fast_dendrogram_distance

# Re-exported unchanged from their module
gamma_coefficients = _vinas.gamma_coefficients
gamma_coef = _vinas.gamma_coef
pearson_correlation = _vinas.pearson_correlation
correlations_list = _vinas.correlations_list
hierarchical_clustering = _vinas.hierarchical_clustering
compare_cophenetic = _vinas.compare_cophenetic
standardize = _vinas.standardize


def gamma_scores(x_real, x_other):
    """S_dist and S_dend between two expression matrices sharing a gene axis.

    Args:
        x_real, x_other: (n_samples, n_genes). Gene ordering must match between
            them, but need not match anyone else's — γ correlates the two
            distance matrices positionally, so the ordering cancels as long as it
            is internally consistent. This is why our numbers are comparable to
            their published 0.920 without recovering their gene permutation.

    Returns:
        dict with s_dist, s_dend, and the two intra-dataset terms.
    """
    g_dx_dz, g_dx_tx, g_dz_tz, g_tx_tz = gamma_coefficients(x_real, x_other)
    return {
        "s_dist": float(g_dx_dz),
        "s_dend": float(g_tx_tz),
        "gamma_real_dendro": float(g_dx_tx),
        "gamma_other_dendro": float(g_dz_tz),
        "sdcc": float((g_dx_tx - g_dz_tz) ** 2),
    }


def sampling_bound(x_real, n_runs=5, seed=0, subset_size=None):
    """Ceiling imposed by sampling noise alone: real vs real, disjoint halves.

    Reimplements gtex_tcga_analysis_2 cells 543-553, which compare x_test against
    an equally-sized random subset of x_train. Any real-vs-synthetic score should
    be read against this, not against 1.0.

    Note a roundtrip has *no* sampling noise — same samples, same order — so it
    should score well ABOVE this bound, not near it.
    """
    rng = np.random.default_rng(seed)
    n = len(x_real)
    half = subset_size or n // 2
    runs = []
    for _ in range(n_runs):
        perm = rng.permutation(n)
        runs.append(gamma_scores(x_real[perm[:half]], x_real[perm[half:2 * half]]))
    keys = runs[0].keys()
    return {
        "mean": {k: float(np.mean([r[k] for r in runs])) for k in keys},
        "std": {k: float(np.std([r[k] for r in runs])) for k in keys},
        "n_runs": n_runs,
    }


def tstr_scores(x_train, y_train, x_test, y_test, n_runs=5, seed=0, max_iter=300):
    """Train-on-synthetic, test-on-real. §5.2.2, absent from the repo.

    Architecture is all the paper specifies: "2 hidden layers of 64 units with
    ReLU activations", averaged over 5 runs. Everything else — optimiser, epochs,
    regularisation — is unstated, so treat the real-trained baseline as the
    meaningful comparator rather than their absolute number.

    Args:
        x_train/y_train: the data under test (synthetic, or roundtripped real)
        x_test/y_test:   held-out real data. y_* are 1-D integer class labels.

    Returns:
        dict with auc / f1_macro / f1_weighted, each mean and std over n_runs.
    """
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import label_binarize

    classes = np.unique(np.concatenate([y_train, y_test]))
    runs = []
    for r in range(n_runs):
        clf = MLPClassifier(hidden_layer_sizes=(64, 64), activation="relu",
                            max_iter=max_iter, random_state=seed + r)
        clf.fit(x_train, y_train)
        proba = clf.predict_proba(x_test)
        pred = clf.classes_[proba.argmax(1)]

        if len(classes) == 2:
            pos = list(clf.classes_).index(classes[1])
            auc = roc_auc_score(y_test, proba[:, pos])
        else:
            y_bin = label_binarize(y_test, classes=clf.classes_)
            auc = roc_auc_score(y_bin, proba, multi_class="ovr", average="macro")
        runs.append({
            "auc": float(auc),
            "f1_macro": float(f1_score(y_test, pred, average="macro")),
            "f1_weighted": float(f1_score(y_test, pred, average="weighted")),
        })
    keys = runs[0].keys()
    return {
        "mean": {k: float(np.mean([r[k] for r in runs])) for k in keys},
        "std": {k: float(np.std([r[k] for r in runs])) for k in keys},
        "n_runs": n_runs,
    }
