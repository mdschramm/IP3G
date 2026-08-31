Start a remote training run on the GCP A100 VM.

The user must specify a module as $ARGUMENTS: `gan`, `diffusion`, or `classifier`.

Default training scripts:
- `gan`: `gan/model.py --train --refresh`
- `diffusion`: `diffusion/diffusion_train.py --mode diagnostic` (the only remote-capable diffusion config — `--mode remote` was removed as stale/unused)
- `classifier`: `classifer/Classifier.py`

Steps:
1. Source `gcloud_helpers` inline — do not check whether it is already sourced. Prefix every Bash call with `source gcloud_helpers &&`.
2. If no module is given in $ARGUMENTS, ask the user which one.
3. Remind the user that preprocessing data must be on the VM. If this is a fresh session (VM was stopped/restarted), they should run `pull_preprocessing_from_gcs` first. Ask if they need to do this.
4. Run `source gcloud_helpers && run_remote "<script>"` with the appropriate script for the chosen module.
5. Immediately stream logs: read the `tail_logs` function body from `gcloud_helpers` and run the underlying `gcloud compute ssh ... sudo docker logs -f` command in a Bash background task, then stream via Monitor. Tell the user to say "stop logs" to stop tailing (the container will keep running).

Note: `run_remote` sets `RUN_MODE=remote` automatically, so outputs go to `output/{module}/remote/` on the VM.
