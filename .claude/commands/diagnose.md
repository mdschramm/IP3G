Run a short diagnostic training run on the GCP VM using the full remote model architecture, then automatically run the diagnostics script against the final checkpoint and clean up.

The diagnostic config (`--mode diagnostic`) uses channels=[128,256,512,512], mixed_precision=True, T=1000, 8000 steps, no intermediate checkpoints or sample images — only the final model is saved. After `run_remote_diagnostics` completes it deletes all outputs except `training_history*.png` and `training_history*.npz`.

Steps:
1. Source `gcloud_helpers` inline. Ask the user if preprocessing data needs to be pulled first (`pull_preprocessing_from_gcs`).
2. Ensure the diagnostic output directories exist on the VM:
   `source gcloud_helpers && gcloud compute ssh mark-test-instance --zone=us-central1-b --command="sudo mkdir -p \$HOME/output/diffusion/diagnostic/checkpoints \$HOME/output/diffusion/diagnostic/samples"`
3. Start the diagnostic training run (detached):
   `source gcloud_helpers && run_remote "diffusion/diffusion_train.py --mode diagnostic"`
4. Stream logs via Monitor — run the underlying `gcloud compute ssh ... sudo docker logs -f --tail 300 ip3g` as a background Bash task. Watch for "Training complete!" in the output.
5. Once training completes, run the diagnostics script against the final checkpoint — this streams output directly here and then cleans up all files except training_history:
   `source gcloud_helpers && run_remote_diagnostics`
   (Uses `output/diffusion/diagnostic/checkpoints/diffusion_model_final.weights.h5` and `--mode diagnostic` by default.)
6. Report the full diagnostic output from step 5, highlighting any ✗ or ⚠ flags.

Note: `run_remote` sets `RUN_MODE=remote` automatically; the diagnostic config overrides output paths regardless of RUN_MODE. The training_history files remain on the VM in `$HOME/output/diffusion/diagnostic/` but are not synced — they are for reference if needed via SSH.
