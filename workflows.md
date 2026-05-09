# Workflows

## Output directories

| Stage | Local | Remote |
|---|---|---|
| Preprocessing | `output/preprocessing/` | `output/preprocessing/` (shared) |
| GAN | `output/gan/local/` | `output/gan/remote/` |
| Classifier | `output/classifier/local/` | `output/classifier/remote/` |
| Diffusion | `output/diffusion/local/` | `output/diffusion/remote/` |

Preprocessing has no local/remote split — it is run once (remotely) and treated as read-only input by all downstream modules.

---

## Local runs (sanity check)

> **M1/M2 Mac note:** `gan/model.py` automatically disables the Metal GPU when `RUN_MODE=local` to avoid a TF Metal XLA incompatibility. If you hit a similar `could not find registered platform` error in other scripts, prefix with `CUDA_VISIBLE_DEVICES=""`.


Run from the project root. `RUN_MODE` defaults to `"local"`. Use low epochs/steps to verify the pipeline runs end-to-end before committing to a full remote run.

```bash
# Preprocessing — run once locally to validate the pipeline
python -m preprocessing.prepare_training_data

# GAN — sanity check (5 epochs)
python -m gan.model --train --refresh --epochs 5
python -m gan.model --generate --samples-per-class 10

# Classifier
python -m classifer.Classifier

# Diffusion
python -m diffusion.diffusion_train --mode local
python -m diffusion.diffusion_sample --mode local \
    --checkpoint output/diffusion/local/checkpoints/diffusion_model_final.weights.h5 \
    --visualize

# Evaluation (requires trained classifier weights in output/classifier/local/)
python -m evaluation.evaluation --mode accuracy
python -m evaluation.evaluation --mode confidence
python -m evaluation.evaluation --mode map
```

---

## Remote runs (full training on A100)

Source `gcloud_helpers` first: `source gcloud_helpers`

`RUN_MODE=remote` is injected automatically by `run_remote()`. Outputs land in `output/{module}/remote/` on the VM, then synced to GCS.

### One-time VM setup

```bash
docker_login_remote                        # authenticate Docker against Artifact Registry
enable_permissions_on_output_dir_for_docker  # create ~/output/* dirs on VM
```

### Push updated code

```bash
push_gcloud_image_fast    # fast rebuild using cached base layer (code changes only)
push_gcloud_image         # full rebuild (use when environment.yml / dependencies change)
```

### Step 1 — Preprocessing (run once; reuse across all model runs)

```bash
USE_GPU=false CONTAINER_NAME=ip3g-preprocess run_remote_preprocess
CONTAINER_NAME=ip3g-preprocess tail_logs   # stream output
push_preprocessing_to_gcs                  # save to GCS when done
```

To reuse preprocessing on a fresh VM session without re-running it:

```bash
pull_preprocessing_from_gcs
```

### Step 2 — GAN training

```bash
run_remote "gan/model.py --train --refresh --epochs 2000"
tail_logs
push_output_to_gcs gan
```

Generate synthetic data after training:

```bash
run_remote "gan/model.py --generate --samples-per-class 100"
tail_logs
push_output_to_gcs gan
```

### Step 3 — Classifier training

```bash
run_remote "classifer/Classifier.py"
tail_logs
push_output_to_gcs classifier
```

### Step 4 — Diffusion training

```bash
run_remote "diffusion/diffusion_train.py --mode remote"
tail_logs
push_output_to_gcs diffusion
```

### Step 5 — Download outputs locally

```bash
download_output_from_gcs    # syncs all of gs://mark-ip3g-data/output/ → ./output/
```

---

## Sample end-to-end workflows

### Workflow A — First time, full pipeline

```bash
push_gcloud_image_fast
enable_permissions_on_output_dir_for_docker

# Preprocessing
USE_GPU=false CONTAINER_NAME=ip3g-preprocess run_remote_preprocess
CONTAINER_NAME=ip3g-preprocess tail_logs
push_preprocessing_to_gcs

# GAN
run_remote "gan/model.py --train --refresh"
tail_logs
push_output_to_gcs gan

# Classifier
run_remote "classifer/Classifier.py"
tail_logs
push_output_to_gcs classifier

# Diffusion
run_remote "diffusion/diffusion_train.py --mode remote"
tail_logs
push_output_to_gcs diffusion

download_output_from_gcs
```

### Workflow B — Iterate on one model (preprocessing already done)

```bash
push_gcloud_image_fast
pull_preprocessing_from_gcs
run_remote "diffusion/diffusion_train.py --mode remote"
tail_logs
push_output_to_gcs diffusion
download_output_from_gcs
```

### Workflow C — Run alongside an existing container

```bash
# The running container is unaffected; use a different name
push_gcloud_image_fast
docker_login_remote
USE_GPU=false CONTAINER_NAME=ip3g-preprocess run_remote_preprocess
CONTAINER_NAME=ip3g-preprocess tail_logs
```

### Workflow D — Debug a crash (capture exit output)

```bash
USE_GPU=false debug_remote_preprocess                       # preprocessing
USE_GPU=false debug_remote_preprocess "gan/model.py --train --refresh --epochs 1"
```

---

## Monitoring and management

```bash
tail_logs                          # stream logs for default container (ip3g)
CONTAINER_NAME=foo tail_logs       # stream logs for named container
stop_ip3g_container                # graceful stop
force_kill_ip3g_containers         # hard kill matching image
force_kill_all_containers          # hard kill everything (caution)
list_registry_images               # show all tagged/untagged images in Artifact Registry
prune_untagged_images              # delete untagged digests to free registry space
```
