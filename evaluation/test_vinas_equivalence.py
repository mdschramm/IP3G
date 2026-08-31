#!/usr/bin/env python
"""
Assert the two performance substitutions in vinas_metrics are exactly equivalent
to the originals they replace.

This is what keeps "we used their metric implementation" an honest claim: the
substitutions exist because their versions do not scale to 18,154 genes, not
because we wanted different numbers. Run it after any submodule bump.

    python -m evaluation.test_vinas_equivalence
"""

import numpy as np

from evaluation import vinas_metrics as vm


def test_upper_diag_list(seed=0):
    rng = np.random.default_rng(seed)
    ok = True
    for n in (5, 50, 400):
        m = rng.normal(size=(n, n))
        m = (m + m.T) / 2
        theirs = vm._ORIGINAL["upper_diag_list"](m)
        ours = vm._fast_upper_diag_list(m)
        same = theirs.shape == ours.shape and np.array_equal(theirs, ours)
        ok &= same
        print(f"  upper_diag_list      n={n:<4} identical={same}  len={len(ours)}")
    return ok


def test_dendrogram_distance(seed=0):
    rng = np.random.default_rng(seed)
    ok = True
    for n_genes in (10, 60, 300):
        x = rng.normal(size=(80, n_genes))
        z = vm.hierarchical_clustering(x)
        theirs = vm._ORIGINAL["dendrogram_distance"](z, condensed=True)
        ours = vm._fast_dendrogram_distance(z, condensed=True)
        same = np.allclose(theirs, ours, rtol=0, atol=0)
        ok &= same
        print(f"  dendrogram_distance  n={n_genes:<4} identical={same}  "
              f"max|diff|={np.abs(theirs - ours).max():.3e}")
    return ok


def test_pearson_clip(seed=0):
    """Clamping must be a no-op on in-range values, and must fix the overshoot."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(60, 40))
    theirs = vm._ORIGINAL["pearson_correlation"](x, x)
    ours = vm._clipped_pearson_correlation(x, x)
    inrange = np.abs(theirs) <= 1.0
    agree = np.array_equal(theirs[inrange], ours[inrange])
    bounded = bool((np.abs(ours) <= 1.0).all())
    print(f"  pearson clip         agrees in-range={agree}  output bounded={bounded}  "
          f"overshoot fixed={int((theirs > 1.0).sum())}")

    # the defaults on both callers must route through the clipped version too
    routed = (vm.correlations_list.__defaults__[0] is vm._clipped_pearson_correlation
              and vm.hierarchical_clustering.__defaults__[0] is vm._clipped_pearson_correlation)
    print(f"  default-arg rebinding correlations_list + hierarchical_clustering={routed}")
    return agree and bounded and routed


def test_no_negative_linkage(seed=0):
    """A linkage built from clipped correlations must be valid for scipy."""
    from scipy.cluster.hierarchy import cophenet, is_valid_linkage
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(40, 150))
    x[:, :20] = x[:, :1]           # perfectly collinear genes -> r == 1 exactly
    z = vm.hierarchical_clustering(x)
    ok = is_valid_linkage(z) and np.all(z[:, 2] >= 0)
    cophenet(z)                    # must not raise
    print(f"  linkage from collinear genes valid={ok}  min dist={z[:, 2].min():.3e}")
    return bool(ok)


def test_gamma_end_to_end(seed=0):
    """A perfect copy must score 1.0; unrelated data must not."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(120, 80))
    same = vm.gamma_scores(x, x.copy())
    diff = vm.gamma_scores(x, rng.normal(size=(120, 80)))
    ok = same["s_dist"] > 0.999 and abs(diff["s_dist"]) < 0.3
    print(f"  gamma identical copy   s_dist={same['s_dist']:.6f} s_dend={same['s_dend']:.6f}")
    print(f"  gamma unrelated data   s_dist={diff['s_dist']:+.4f}")
    return ok


if __name__ == "__main__":
    print("Verifying vinas_metrics substitutions against the originals\n")
    results = [
        ("upper_diag_list", test_upper_diag_list()),
        ("dendrogram_distance", test_dendrogram_distance()),
        ("pearson clip", test_pearson_clip()),
        ("no negative linkage", test_no_negative_linkage()),
        ("gamma end-to-end", test_gamma_end_to_end()),
    ]
    print()
    for name, passed in results:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    raise SystemExit(0 if all(p for _, p in results) else 1)
