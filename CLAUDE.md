# IP3G — GTEx Gene Expression Image Generation

## Project Overview
Generates synthetic gene expression images from GTEx tissue samples using GAN and diffusion models. Input: normalized RNA-seq counts + tissue phenotype metadata. Output: 128×128 images representing per-tissue expression profiles.

Pipeline: GTEx raw data → preprocessing → image representation → GAN or diffusion training → evaluation

## Module Structure

| Module | Entry point | Purpose |
|--------|-------------|---------|
| `preprocessing/` | `prepare_training_data.py` | Normalize GTEx data → `.npy` arrays + images |
| `gan/` | `model.py` | WINFOGAN training and generation |
| `classifer/` | `Classifier.py` | Tissue-type classifier (intentional typo — leave as-is, all imports match) |
| `diffusion/` | `diffusion_train.py` | DDPM U-Net training; `diffusion_sample.py` for inference |
| `evaluation/` | `evaluation.py` | Accuracy, confidence, mAP metrics |

## Data Flow

```
GTEx files (GTEX_phenotype, gtex_RSEM_*, gtex_gene_*)
  └─ preprocessing/ ──► output/preprocessing/          # shared, read-only for all models
       └─ gan/          ──► output/gan/{local|remote}/
       └─ diffusion/    ──► output/diffusion/{local|remote}/
       └─ classifer/    ──► output/classifier/{local|remote}/
            └─ evaluation/ reads classifier + model outputs
```

## Running Locally (sanity checks, low epoch counts)

Run from project root with the `dataexplr` conda environment:

```bash
python -m preprocessing.prepare_training_data
python -m gan.model --train --refresh --epochs 5
python -m gan.model --generate --samples-per-class 10
python -m classifer.Classifier
python -m diffusion.diffusion_train --mode local
python -m diffusion.diffusion_sample --mode local --checkpoint output/diffusion/local/checkpoints/diffusion_model_final.weights.h5 --visualize
python -m evaluation.evaluation --mode accuracy
python -m evaluation.evaluation --mode map
```

## Running Remotely (A100 on GCP)

Source `gcloud_helpers` first — the QUICK REFERENCE block at the top of that file lists the exact commands in order. See `workflows.md` for full step-by-step walkthroughs including first-time VM setup.

Slash commands available in Claude Code: `/deploy`, `/train`, `/logs`, `/sync`, `/eval`

## Key Configuration

- `RUN_MODE` env var: `"local"` (default) or `"remote"` — controls which output subdirectory is used
- Output dirs: `output/{module}/{local|remote}/`
- Preprocessing output: `output/preprocessing/` (no local/remote split — shared input)
- Mixed precision: disabled locally (M1 Metal), enabled remotely (A100)
- `jit_compile=False` on GAN compile — disables XLA JIT; required on both Metal and A100. Enabling it under mixed_float16 on A100 causes mode collapse at ~epoch 663 (XLA op fusion changes float16 numerical trajectory).

## Important Notes

- The folder name `classifer/` is a typo but is consistent everywhere (imports, Dockerfile, tests). Do not rename.
- `SparseSelfAttention` in `diffusion/diffusion_model.py` uses multiplicative top-k gating (not bool masking) to avoid a Metal BroadcastTo kernel bug with bool tensors.
- `VM_OUTPUT_BASE='$HOME/output'` in `gcloud_helpers` uses **single quotes** intentionally — prevents local `$HOME` expansion so `$HOME` resolves to `/home/mschramm` on the VM.

## What to Ignore

- `legacy/` — old notebooks and deprecated scripts from before the module refactor. Do not read or modify.
- `legacy/assistantplans/` — Claude planning documents from earlier sessions, outdated.
- `output/` — generated artifacts, not tracked in git.
- `__pycache__/` — Python bytecode cache.
