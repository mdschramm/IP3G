Run evaluation on trained models locally.

Valid modes passed as $ARGUMENTS:
- `accuracy` — evaluate classifier on real preprocessed data (train/val split)
- `confidence` — per-class confidence of classifier on real data
- `map` — GAN: map unsupervised latent classes → phenotypes
- `map diffusion` — diffusion class fidelity: for each class C, what fraction of class-C-conditioned images are classified as C?

Default mode if none given: `accuracy`.

Commands to run:
- `accuracy`:  `python -m evaluation.evaluation --mode accuracy`
- `confidence`: `python -m evaluation.evaluation --mode confidence`
- `map` (GAN): `python -m evaluation.evaluation --mode map --source gan`
- `map` (diffusion): `python -m evaluation.evaluation --mode map --source diffusion`
  - Reads from `output/diffusion/{RUN_MODE}/samples/diffusion_synthetic_expressions_w3.0.npy`
  - If that file is missing, generate samples first with `/generate-samples` or:
    `python -m diffusion.diffusion_sample --mode local --checkpoint <path> --generate-dataset`

Prerequisites to check before running:
- `output/preprocessing/` must exist with `.npy` files
- `output/classifier/local/` or `output/classifier/remote/` must have `classifier_weights_only.keras`
- For `map --source gan`: `output/gan/{mode}/synthetic_resized_expressions.npy` and `synthetic_latent_classes.npy`
- For `map --source diffusion`: `output/diffusion/{mode}/samples/diffusion_synthetic_expressions_w3.0.npy`

If any prerequisite is missing, tell the user what to run first. Run `/sync` first if you need the latest remote outputs locally.
