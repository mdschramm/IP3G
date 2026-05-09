Generate a synthetic dataset from the trained diffusion model. This is required before running `/eval map diffusion`.

The user may pass options as $ARGUMENTS (e.g. `remote 3.0 50` for mode, guidance scale, and samples per class).

Defaults:
- mode: `local`
- guidance scale: `3.0`
- samples per class: `100`

Checkpoint path is auto-detected from the mode:
- local:  `output/diffusion/local/checkpoints/diffusion_model_ema.weights.h5`
- remote: `output/diffusion/remote/checkpoints/diffusion_model_ema.weights.h5`

Steps:
1. Parse $ARGUMENTS for optional overrides (mode, guidance scale, samples per class).
2. Confirm the checkpoint file exists at the expected path. If not, list available checkpoints in the checkpoint dir and let the user pick one.
3. Run:
   ```
   python -m diffusion.diffusion_sample \
     --mode <mode> \
     --checkpoint <checkpoint_path> \
     --generate-dataset \
     --samples-per-class <n> \
     --guidance-scale <w> \
     --output-dir output/diffusion/<mode>/samples
   ```
4. Report the saved file paths and shapes so the user knows what to pass to `/eval map diffusion`.
