Summarize all available Claude Code slash commands for this project.

Print the following reference (do not run any tools, just output this text):

---

## IP3G Slash Commands

| Command | Usage | What it does |
|---------|-------|--------------|
| `/deploy` | `/deploy [--full\|--base]` | Push code to GCP. Default: fast build (code only, ~1-2 min). `--full` rebuilds everything including deps. `--base` rebuilds only the dependency layer. |
| `/train` | `/train <module>` | Start a remote training run on the A100 VM. Modules: `gan`, `diffusion`, `classifier`. Tails logs automatically. |
| `/diagnose` | `/diagnose` | Run a 1000-step diagnostic training run on the A100 using the full remote architecture (~10 min). Streams logs, syncs outputs, then runs `diffusion/diagnostics.py` locally and reports results. Use before a full run to catch config/architecture issues early. |
| `/logs` | `/logs [container-name]` | Stream logs from the running container (default: `ip3g`). |
| `/sync` | `/sync [module]` | Push VM outputs → GCS → download to local `./output/`. Module optional (`gan`, `diffusion`, `classifier`); omit to sync all. |
| `/generate-samples` | `/generate-samples [mode] [guidance_scale] [samples_per_class]` | Generate a synthetic dataset from the trained diffusion model. Required before running `/eval map diffusion`. Defaults: local, w=3.0, 100 samples/class. |
| `/eval` | `/eval [mode]` | Run evaluation locally. Modes: `accuracy`, `confidence`, `map` (GAN), `map diffusion` (diffusion class fidelity). |
| `/help` | `/help` | Show this reference. |

## Typical Remote Workflow

```
source gcloud_helpers           # load helpers (once per terminal session)

/deploy                         # push latest code (~1-2 min)
/diagnose                       # 1000-step sanity check on A100 (~10 min) — validate before full run
/train diffusion                # start full 75k-step training + tail logs (Ctrl+C to stop tailing)

# ... wait for training to finish ...

/sync diffusion                 # VM → GCS → local ./output/diffusion/remote/
/generate-samples remote        # generate synthetic dataset from trained model
/eval map diffusion             # evaluate class fidelity
```

## Diagnostic Modes

`diffusion_train.py` accepts three `--mode` values:
- `local` — small architecture (channels=[64,128,256]), 3000 steps, Mac M2
- `diagnostic` — full remote architecture, 1000 steps, mixed_precision=True (fast remote sanity check)
- `remote` — full remote architecture, 75000 steps, A100 full training run

## Tips

- **Re-auth**: If Docker pull fails with "unauthorized", run `docker_login_remote`.
- **Preprocessing**: Run once, save with `push_preprocessing_to_gcs`, reuse with `pull_preprocessing_from_gcs`.
- **Parallel containers**: Set `CONTAINER_NAME=my-name` before `run_remote`/`tail_logs` to run alongside an existing container.
- **Diagnostic script**: Run `python -m diffusion.diagnostics --help` for standalone usage against any checkpoint.
- **Full reference**: See `workflows.md` for complete step-by-step walkthroughs.
