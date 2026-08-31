"""
Loader for the combined TCGA + GTEx corpus published by Wang et al. (2018).

Source: https://github.com/mskcc/RNAseqDB  (data/normalized/, 52 gzipped TSVs)
Paper:  https://www.nature.com/articles/sdata201861

This is the RNAseqDB counterpart to the GTEx-specific loaders in
preprocess_data.py. It produces the same two things the rest of the pipeline
needs — an (N_samples, N_genes) expression matrix and a per-sample label table —
so everything downstream (t-SNE, F-statistic ordering, image mapping,
reconstruction) is reused unchanged.

THREE THINGS DIFFER FROM THE GTEx PATH, and each is load-bearing:

1. The 52 files do NOT share a gene list. They carry exactly 12 distinct lists
   (18,764-19,969 genes) which map one-to-one onto the 12 non-empty ComBat
   clusters in the upstream tissue-conf.txt: ComBat ran per cluster and dropped
   genes with no variance within that cluster's samples. We take the
   INTERSECTION (18,154 genes), matching Vinas et al. (Bioinformatics 2022).

   The union is tempting and wrong: zero-filling absent genes would make
   missingness systematically correlated with tissue, since it was produced by
   per-cluster ComBat. "Gene X is exactly 0 across every liver sample" then
   becomes a perfect tissue tell that a generative model will happily learn and
   a classifier will score as tissue accuracy.

2. Values are linear FPKM, not log2. The GTEx pipeline relies on Xena having
   already applied log2(x+1) upstream; here we apply it ourselves.

3. The published FPKM contains ComBat blow-ups — 884 cells above 1e6, topping
   out at 4.8e11 (MAGEA4 in one lung GTEx sample). These cluster in the
   small-N groups (cervix-gtex has n=11) and in mitochondrial genes. Left
   alone, a single artifact sets channel_scales[k] downstream and compresses
   every real value into the bottom of the range, so we clip after the log.

USAGE:
    from preprocessing.rnaseqdb_loader import load_rnaseqdb

    X, genes, labels = load_rnaseqdb()
    # X:      (9147, 18154) float32, log2(FPKM+1) clipped to [0, LOG2_CLIP]
    # genes:  DataFrame with Hugo_Symbol / Entrez_Gene_Id, row-aligned to X's columns
    # labels: DataFrame with sample_id, tissue, condition, subtype, source
"""

from __future__ import annotations

import glob
import gzip
import os
import urllib.request

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- constants --

RNASEQDB_RAW_BASE = (
    "https://raw.githubusercontent.com/mskcc/RNAseqDB/master/data/normalized"
)
RNASEQDB_API_URL = (
    "https://api.github.com/repos/mskcc/RNAseqDB/contents/data/normalized"
)
DEFAULT_DATA_DIR = "data/rnaseqdb/normalized"

# Verified against the published files; assert rather than trust.
EXPECTED_FILES = 52
EXPECTED_GENES = 18_154
EXPECTED_SAMPLES = 9_147

# log2(FPKM+1) ceiling. Measured on an 8-file / 2,488-sample subset spanning
# both sources and five clusters: p99=14.08, p99.9=17.04, p99.99=19.62,
# max=38.81. Clipping at 20 touches 0.0068% of values (pure ComBat artifacts)
# and lands std ~= 0.177, within 4% of the GTEx pipeline's sigma_data 0.1709.
LOG2_CLIP = 20.0

# Esophageal carcinoma has no subregion in TCGA, but GTEx splits esophagus into
# three. Carcinoma arises from the mucosal epithelium, so esca maps there.
# Vinas et al. effectively used esophagus_mus (their tcga->gtex dict is built
# last-wins over the three conf rows, which is an accident rather than a
# choice); set this to "esophagus_mus" to reproduce them exactly. 194 samples.
ESCA_TISSUE = "esophagus_muc"

# TCGA cancer code -> anatomical site. 1:1 from the TCGA side except esca.
TCGA_TO_TISSUE = {
    "prad": "prostate", "blca": "bladder", "brca": "breast", "thca": "thyroid",
    "stad": "stomach",  "luad": "lung",    "lusc": "lung",   "lihc": "liver",
    "chol": "liver",    "kirc": "kidney",  "kirp": "kidney", "kich": "kidney",
    "coad": "colon",    "read": "colon",   "ucec": "uterus", "ucs": "uterus",
    "cesc": "cervix",   "hnsc": "salivary",
    "esca": ESCA_TISSUE,
}

ATTRIBUTES = ("tissue", "condition", "subtype", "source")

# Subtype value for every normal sample. Deliberately distinct from the
# classifier-free-guidance null token, which means "any subtype" and lives at
# its own index outside the vocabulary.
SUBTYPE_NONE = "none"


# ----------------------------------------------------------------- download --

def _remote_file_names() -> list[str]:
    """List the .txt.gz files in the upstream data/normalized directory."""
    import json

    with urllib.request.urlopen(RNASEQDB_API_URL, timeout=60) as resp:
        entries = json.load(resp)
    return sorted(e["name"] for e in entries if e["name"].endswith(".txt.gz"))


def download_rnaseqdb(data_dir: str = DEFAULT_DATA_DIR, force: bool = False) -> list[str]:
    """Fetch the 52 normalized matrices, skipping any already present.

    Args:
        data_dir: destination directory (created if absent)
        force: re-download even when the file exists

    Returns:
        Sorted list of local .txt.gz paths.
    """
    os.makedirs(data_dir, exist_ok=True)
    existing = sorted(glob.glob(os.path.join(data_dir, "*.txt.gz")))
    if len(existing) == EXPECTED_FILES and not force:
        print(f"  {EXPECTED_FILES} files already present in {data_dir}")
        return existing

    names = _remote_file_names()
    print(f"  {len(names)} files upstream; downloading into {data_dir}")
    for i, name in enumerate(names, 1):
        dest = os.path.join(data_dir, name)
        if os.path.exists(dest) and not force:
            continue
        urllib.request.urlretrieve(f"{RNASEQDB_RAW_BASE}/{name}", dest)
        print(f"    [{i:>2}/{len(names)}] {name}")

    files = sorted(glob.glob(os.path.join(data_dir, "*.txt.gz")))
    if len(files) != EXPECTED_FILES:
        raise RuntimeError(
            f"Expected {EXPECTED_FILES} files in {data_dir}, found {len(files)}"
        )
    return files


# ------------------------------------------------------------- file parsing --

def parse_filename(path: str) -> tuple[str, str]:
    """Split a matrix filename into its tissue/cancer code and its kind.

    'blca-rsem-fpkm-tcga-t.txt.gz' -> ('blca', 'tcga-t')
    'liver-rsem-fpkm-gtex.txt.gz'  -> ('liver', 'gtex')
    """
    base = os.path.basename(path)
    for suffix in (".txt.gz", ".txt"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    parts = base.split("-")
    # {code}-rsem-fpkm-{kind...}
    return parts[0], "-".join(parts[3:])


def _read_header(path: str) -> list[str]:
    """Sample IDs from a matrix header, without decompressing the whole file."""
    with gzip.open(path, "rt") as fh:
        return fh.readline().rstrip("\n").split("\t")[2:]


def _tcga_condition_from_barcode(barcode: str) -> str:
    """Read tumor/normal out of TCGA barcode field 4.

    TCGA sample type codes: 01-09 tumor, 10-19 normal, 20-29 control.
    e.g. TCGA-GU-A42P-01A-11R-A23W-07 -> '01' -> tumor
    """
    fields = barcode.split("-")
    if len(fields) < 4:
        raise ValueError(f"Malformed TCGA barcode: {barcode!r}")
    code = int(fields[3][:2])
    if code < 10:
        return "tumor"
    if code < 20:
        return "normal"
    raise ValueError(f"Unexpected TCGA sample type {code:02d} in {barcode!r}")


def label_rows(path: str, sample_ids: list[str]) -> list[dict]:
    """Derive the four label attributes for every sample in one matrix file.

    Tissue, condition, subtype and source all come from the filename. For TCGA
    files the condition is independently re-derived from each barcode and the
    two must agree — a free consistency check on the upstream release.
    """
    code, kind = parse_filename(path)

    if kind == "gtex":
        tissue, condition, source, subtype = code, "normal", "gtex", SUBTYPE_NONE
    elif kind in ("tcga", "tcga-t"):
        if code not in TCGA_TO_TISSUE:
            raise KeyError(f"No tissue mapping for TCGA code {code!r} ({path})")
        tissue = TCGA_TO_TISSUE[code]
        source = "tcga"
        condition = "tumor" if kind == "tcga-t" else "normal"
        subtype = code if condition == "tumor" else SUBTYPE_NONE
    else:
        raise ValueError(f"Unrecognized file kind {kind!r} for {path}")

    rows = []
    for sid in sample_ids:
        if source == "tcga":
            from_barcode = _tcga_condition_from_barcode(sid)
            if from_barcode != condition:
                raise ValueError(
                    f"{os.path.basename(path)}: filename says {condition} but "
                    f"barcode {sid} says {from_barcode}"
                )
        rows.append({
            "sample_id": sid,
            "tissue": tissue,
            "condition": condition,
            "subtype": subtype,
            "source": source,
        })
    return rows


# ------------------------------------------------------------ gene handling --

def gene_intersection(files: list[str], verbose: bool = True) -> pd.MultiIndex:
    """Genes present in every one of the 52 matrices, in canonical sorted order.

    Sorting is not cosmetic: pandas' Index.intersection does not guarantee a
    stable order, and this order fixes the gene axis of data.npy, the t-SNE
    input, and every gene->pixel mapping downstream. An unsorted index would
    silently reshuffle artifacts between runs.
    """
    index = None
    sizes = []
    for path in files:
        ids = pd.read_csv(path, sep="\t", usecols=[0, 1], compression="gzip")
        idx = pd.MultiIndex.from_frame(ids)
        if idx.has_duplicates:
            raise ValueError(f"Duplicate (symbol, entrez) pairs in {path}")
        sizes.append(len(idx))
        index = idx if index is None else index.intersection(idx)

    index = index.sort_values()
    if verbose:
        print(f"  per-file gene counts: min={min(sizes)} max={max(sizes)} "
              f"({len(set(sizes))} distinct sizes across {len(files)} files)")
        print(f"  intersection: {len(index)} genes")
    return index


# ------------------------------------------------------------------ loading --

def load_rnaseqdb(
    data_dir: str = DEFAULT_DATA_DIR,
    clip: float = LOG2_CLIP,
    download: bool = True,
    verbose: bool = True,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Load the combined corpus as an expression matrix plus a label frame.

    Args:
        data_dir: directory holding the 52 *.txt.gz matrices
        clip: upper bound applied after log2(FPKM+1); None disables clipping
        download: fetch missing files from GitHub first
        verbose: print progress

    Returns:
        X:      (N_samples, N_genes) float32, log2(FPKM+1) clipped to [0, clip]
        genes:  (N_genes, 2) DataFrame — Hugo_Symbol, Entrez_Gene_Id
        labels: (N_samples, 5) DataFrame — sample_id + the four attributes
    """
    if download:
        files = download_rnaseqdb(data_dir)
    else:
        files = sorted(glob.glob(os.path.join(data_dir, "*.txt.gz")))
    if len(files) != EXPECTED_FILES:
        raise RuntimeError(f"Expected {EXPECTED_FILES} matrices, found {len(files)}")

    if verbose:
        print("\n[1/3] Building gene intersection...")
    gene_index = gene_intersection(files, verbose=verbose)
    if len(gene_index) != EXPECTED_GENES:
        raise RuntimeError(
            f"Gene intersection is {len(gene_index)}, expected {EXPECTED_GENES}. "
            "The upstream release may have changed."
        )

    if verbose:
        print("\n[2/3] Building label frame...")
    label_records, per_file_counts = [], []
    for path in files:
        ids = _read_header(path)
        per_file_counts.append(len(ids))
        label_records.extend(label_rows(path, ids))
    labels = pd.DataFrame.from_records(label_records)

    n_samples = len(labels)
    if n_samples != EXPECTED_SAMPLES:
        raise RuntimeError(f"Got {n_samples} samples, expected {EXPECTED_SAMPLES}")
    if labels["sample_id"].duplicated().any():
        dupes = labels.loc[labels["sample_id"].duplicated(), "sample_id"].tolist()
        raise ValueError(f"Duplicate sample IDs across files: {dupes[:5]}")
    if verbose:
        print(f"  {n_samples} samples, barcode/filename condition check passed")

    # Fill column blocks in place rather than concatenating 52 frames, which
    # would hold the whole corpus in memory twice at peak.
    if verbose:
        print(f"\n[3/3] Reading expression into ({n_samples}, {len(gene_index)}) float32...")
    X = np.empty((len(gene_index), n_samples), dtype=np.float32)
    col = 0
    for i, path in enumerate(files, 1):
        df = pd.read_csv(path, sep="\t", index_col=[0, 1], compression="gzip")
        block = df.loc[gene_index].to_numpy(dtype=np.float32, copy=False)
        X[:, col:col + block.shape[1]] = block
        col += block.shape[1]
        if verbose and (i % 10 == 0 or i == len(files)):
            print(f"    {i:>2}/{len(files)} files, {col}/{n_samples} samples")
    assert col == n_samples, f"filled {col} columns, expected {n_samples}"

    np.log2(X + 1.0, out=X)
    if clip is not None:
        n_clipped = int((X > clip).sum())
        np.clip(X, 0.0, clip, out=X)
        if verbose:
            print(f"  log2(FPKM+1) applied; clipped {n_clipped:,} values "
                  f"({100 * n_clipped / X.size:.4f}%) at {clip}")

    genes = gene_index.to_frame(index=False)
    X = np.ascontiguousarray(X.T)  # -> (samples, genes)

    if verbose:
        print(f"\n  X: {X.shape} {X.dtype}  range [{X.min():.3f}, {X.max():.3f}]  "
              f"mean {X.mean():.3f}")
    return X, genes, labels
