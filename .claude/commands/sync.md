Sync training outputs from the GCP VM → GCS → local machine.

The user may optionally pass a module name as $ARGUMENTS: `gan`, `diffusion`, or `classifier`. If none is given, sync all modules.

Steps:
1. Source `gcloud_helpers` inline before each command — do not check whether it is already sourced. Prefix every Bash call with `source gcloud_helpers &&`.
2. Push VM outputs to GCS:
   - With module: `source gcloud_helpers && push_output_to_gcs <module>`
   - Without module: `source gcloud_helpers && push_output_to_gcs` (syncs gan, classifier, diffusion)
3. Download from GCS to local `./output/`: `source gcloud_helpers && download_output_from_gcs`
4. Report what was downloaded and where to find it (e.g., `output/diffusion/remote/checkpoints/`).

Tip: Run this after `tail_logs` shows training has finished or reached a checkpoint you want to inspect locally.
