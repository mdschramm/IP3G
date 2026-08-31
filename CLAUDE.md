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
| `diffusion/` | `diffusion_train.py` | EDM2 (Karras et al.) U-Net training; `diffusion_edm_sample.py` for inference — see `diffusion/ARCHITECTURE.md` for full model details |
| `evaluation/` | `evaluation.py` | Accuracy, confidence, mAP metrics on trained models; also home to preprocessing-analysis scripts (see below) |

## Data Flow

```
GTEx files (GTEX_phenotype, gtex_RSEM_*, gtex_gene_*)
  └─ preprocessing/ ──► output/preprocessing/          # shared, read-only for all models
       └─ gan/          ──► output/gan/{local|remote}/
       └─ diffusion/    ──► output/diffusion/{local|diagnostic}/     # no "remote" mode — see Key Configuration
       └─ classifer/    ──► output/classifier/{local|remote}/
            └─ evaluation/ reads classifier + model outputs
```

Preprocessing paths and image shape (width/height/channels) are centralized in
`preprocessing/artifact_paths.py` (`PreprocessingConfig` / `DEFAULT_CONFIG`) — every module above
imports from it rather than hardcoding paths. The default config (128×128×16) resolves to the
existing unsuffixed `output/preprocessing/` paths; any other width/height/channels combination
resolves to a tagged subdirectory instead — `output/preprocessing/{W}x{H}x{C}/` and
`output/evaluation/{W}x{H}x{C}/` — so alternate configurations (e.g. built for a preprocessing
comparison) never collide with or overwrite the production artifacts.

## Running Locally (sanity checks, low epoch counts)

Run from project root with the `dataexplr` conda environment:

```bash
python -m preprocessing.prepare_training_data
python -m gan.model --train --refresh --epochs 5
python -m gan.model --generate --samples-per-class 10
python -m classifer.Classifier
python -m diffusion.diffusion_train --mode local
python -m diffusion.diffusion_edm_sample --mode local --checkpoint output/diffusion/local/checkpoints/diffusion_model_final.weights.h5 --generate-dataset --samples-per-class 10
python -m evaluation.evaluation --mode accuracy
python -m evaluation.evaluation --mode map
```

Before a real local training run (especially diffusion, which loads the full feature array into
RAM), sanity-check that it fits your machine's memory rather than assuming it does — see
"Local memory feasibility" below.

## Evaluating and Comparing Preprocessing Configurations

`preprocessing/prepare_training_data.py` accepts `--width/--height/--channels` (default
128/128/16, i.e. production) to build an alternate-resolution/channel dataset — used, for
example, to compare the production 16-channel scheme against single-channel candidates at other
resolutions:

```bash
python -m preprocessing.prepare_training_data --width 256 --height 256 --channels 1
```

The t-SNE embedding and gene-importance ranking are size/channel-independent and are always
reused from the shared `output/preprocessing/` artifacts — only the size-dependent outputs
(resized images, occupancy mask, gene→pixel→channel map, per-channel scales) get rebuilt, into
the config's tagged subdirectory (see Data Flow above).

Two evaluation scripts, both parameterized the same way, analyze the result:

```bash
python -m evaluation.pixel_collision_analysis --width 256 --height 256 --channels 1        # gene/pixel collision counts + discrimination power
python -m evaluation.gene_reconstruction_analysis --width 256 --height 256 --channels 1     # reconstruction fidelity; exact vs. collision-averaged gene accounting
```

`evaluation/diffusion_preprocessing_eval.py` (no size/channel args — always runs against
`DEFAULT_CONFIG`) visualizes the EDM2 forward-noising process on real preprocessed samples,
independent of any trained model.

## Running Remotely (A100 on GCP)

Source `gcloud_helpers` first — the QUICK REFERENCE block at the top of that file lists the exact commands in order. See `workflows.md` for full step-by-step walkthroughs including first-time VM setup.

Slash commands available in Claude Code: `/deploy`, `/train`, `/logs`, `/sync`, `/eval`

## Key Configuration

- `RUN_MODE` env var: `"local"` (default) or `"remote"` — controls the output subdirectory for **gan** and **classifer**. Diffusion does not read `RUN_MODE`; it selects its config via `diffusion_train.py --mode {local,diagnostic}` instead (`diffusion_config.py: get_config()`).
- Output dirs: `output/{module}/{local|remote}/` for gan/classifier; `output/diffusion/{local|diagnostic}/` for diffusion. A `remote` diffusion config existed at one point but was unused/stale and was removed — `diagnostic` (full remote architecture, bounded step count) is what actually runs on the A100.
- Preprocessing output: `output/preprocessing/` for the default (128×128×16) config; other configs use tagged subdirectories (see Data Flow above). No local/remote split — shared, read-only input for all downstream modules.
- Mixed precision: GAN disables it locally and enables it remotely (`RUN_MODE` check in `gan/model.py`). Diffusion deliberately enables it **locally too** (`CONFIG_LOCAL['mixed_precision'] = True` in `diffusion_config.py`) so Metal-specific FP16 issues surface on a cheap local run before they cost A100 hours — this is intentional, not a bug.
- `jit_compile=False` on GAN compile — disables XLA JIT; required on both Metal and A100. Enabling it under mixed_float16 on A100 causes mode collapse at ~epoch 663 (XLA op fusion changes float16 numerical trajectory).

### Local memory feasibility

`diffusion_train.py` loads the full feature array into RAM as one `np.load(...).astype(np.float32, copy=False)`
call — the `copy=False` matters: without it, `.astype()` unconditionally copies even when the
array is already float32 on disk, transiently doubling that array's footprint (~8.2GB → ~16.5GB
for the current 128×128×16 dataset) before Python can free the original. On a 16GB Mac that's a
real OOM/thrash risk. Before trusting a new local config (a different resolution/channel count,
or a bigger batch size), don't just estimate — run a short bounded probe: copy the real config,
drop `num_steps` to ~5 and `sample_interval`/`diag_interval`/`save_interval` to something huge
so periodic sampling/diagnostics don't fire, point `checkpoint_dir`/`sample_dir` at a scratch
directory, and call `diffusion_train.train(config)` directly under `/usr/bin/time -l` (macOS) to
read real peak RSS — cheap (a few minutes), and it exercises the actual data-loading, model-build,
and train-step code path instead of a hand-estimated one.

## Important Notes

- The folder name `classifer/` is a typo but is consistent everywhere (imports, Dockerfile, tests). Do not rename.
- `SparseSelfAttention` in `diffusion/diffusion_model.py` uses multiplicative top-k gating (not bool masking) to avoid a Metal BroadcastTo kernel bug with bool tensors.
- Keras 3's Functional API rejects raw `tf.*` ops applied directly to a `KerasTensor` placeholder outside a `Layer.call()` (e.g. `tf.cast(mask_input, ...)` at model-definition time) — use `keras.ops.*` instead (e.g. `keras.ops.cast`). This previously blocked `build_unet()` from constructing at any resolution; fixed in `diffusion_model.py`'s occupancy-mask conditioning.
- `VM_OUTPUT_BASE='$HOME/output'` in `gcloud_helpers` uses **single quotes** intentionally — prevents local `$HOME` expansion so `$HOME` resolves to `/home/mschramm` on the VM.
- Shell state (e.g. `conda activate dataexplr`) does not persist across separate non-interactive tool invocations in some automation contexts (including Claude Code's Bash tool). When commands run each in their own subshell, prefer the explicit interpreter path — `~/miniconda3/envs/dataexplr/bin/python` — over relying on activation.

## What to Ignore

- `legacy/` — old notebooks and deprecated scripts from before the module refactor. Do not read or modify.
- `legacy/assistantplans/` — Claude planning documents from earlier sessions, outdated.
- `output/` — generated artifacts, not tracked in git.
- `__pycache__/` — Python bytecode cache.
