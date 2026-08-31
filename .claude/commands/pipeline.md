Run the full remote workflow end-to-end: deploy the current code, get the GCP VM running,
pull the new image, and kick off a training run — pausing to confirm with the user at every
step that costs money or needs something only they can do.

$ARGUMENTS is `[module] [dataset]`, both optional and order-independent enough to be obvious:

- **module**: `gan`, `diffusion` (default), `classifier`, `classifier-small`, `fidelity`, or
  `synthetic`. For `diffusion`, training always uses `--mode diagnostic` (the only
  remote-capable diffusion config — see `diffusion/diffusion_config.py`). `fidelity` is the
  M3.5 encoding gate, not a training run. `synthetic` is the M6 train-on-synthetic comparison
  and is a multi-step sequence, not a single run.
- **dataset**: `gtex` (default) or `rnaseqdb` — the combined TCGA+GTEx corpus.

Dataset selection is an **environment variable, not a script flag**: export `RUN_DATASET`
before calling the helper and every entry point picks it up through
`preprocessing/artifact_paths.py`, exactly as `RUN_MODE` already selects local vs remote.
No module takes a `--dataset` argument and none should — that is the whole point of the
design. `RUN_DATASET` is threaded into the container by `run_remote`, `run_remote_fg` and
`run_remote_preprocess` in `gcloud_helpers`, and it steers both the preprocessing artifacts
that get read and the output directory that gets written:

    RUN_DATASET=gtex      ->  output/preprocessing/            output/classifier/remote/
    RUN_DATASET=rnaseqdb  ->  output/preprocessing/rnaseqdb/   output/classifier/remote/rnaseqdb/

Because the non-default corpus nests *inside* the directories that are already bind-mounted
and already rsynced recursively, adding a dataset needs no new mounts and no `/sync` changes.

If the user asks for `rnaseqdb` and `output/preprocessing/rnaseqdb/` does not yet exist on the
VM, the artifacts have to be built there first — do **not** offer to upload them from local
(~10.3 GB over a home connection). Instead run
`source gcloud_helpers && RUN_DATASET=rnaseqdb run_remote_preprocess "preprocessing/prepare_rnaseqdb_data.py"`,
which downloads its own 365 MB of source matrices inside GCP, then
`push_preprocessing_to_gcs`. Tell the user this is a one-time ~5 minute step and confirm before
starting it.

This command always sources `gcloud_helpers` inline before every step — do not check whether
it's already sourced, and never hardcode the instance name or zone; always use `$INSTANCE_NAME`
/ `$GCLOUD_INSTANCE_ZONE` from `gcloud_helpers` so this stays correct if the VM is ever renamed
(the instance is currently `mark-diffusion-test` in `us-central1-b`, but don't hardcode that).

Steps:

1. **Deploy.** Run `source gcloud_helpers && push_gcloud_image_fast`. This only rebuilds the
   runtime layer (code changes) and doesn't need the VM running. If it fails, stop here, show
   the last ~50 lines of output, and suggest `docker_login_remote` (auth) or `/deploy --full`
   (if `environment.yml`/`requirements.txt` changed) — do not proceed to VM steps.

2. **Check VM status.** Run
   `source gcloud_helpers && gcloud compute instances describe $INSTANCE_NAME --zone=$GCLOUD_INSTANCE_ZONE --format='value(status)'`.

3. **If already RUNNING:** tell the user and skip to step 5.
   **If TERMINATED/STOPPED:** this is a real billing decision — ask the user via AskUserQuestion
   whether to start it now (options: start it now / I'll start it myself — tell me when it's up /
   cancel). Do not start it without an explicit yes on this call, every time, even if a prior
   run of this same pipeline started it.
   - If the user says to start it: run
     `source gcloud_helpers && gcloud compute instances start $INSTANCE_NAME --zone=$GCLOUD_INSTANCE_ZONE`.
   - If the user says they'll start it themselves: stop and wait for them to tell you it's up
     before continuing — don't poll for this, they know when they've done it.

4. **Wait for SSH to come up** (only after a start you triggered — a self-reported start from
   the user can proceed straight to step 5). Boot + sshd takes a little while after `instances
   start` reports success. Use Monitor with an until-loop (do not use bare `sleep`) — something
   like: `until gcloud compute ssh $INSTANCE_NAME --zone=$GCLOUD_INSTANCE_ZONE --command="echo ready" 2>/dev/null; do sleep 10; done`,
   capped around 5 minutes. If it doesn't come up in that window, stop and tell the user rather
   than continuing to hang.

5. **Pull the new image onto the VM.** Run `source gcloud_helpers && set_up_docker_image`
   (re-auths Docker, pulls the latest image, mounts the GCS bucket — same as `/deploy` step 2).
   This is safe to re-run on an already-configured VM.

6. **Kick off the run (detached).** Prefix every command with `RUN_DATASET=<dataset>` (omit it,
   or use `gtex`, for the default corpus). Based on the module from $ARGUMENTS:
   - `diffusion` (default): `source gcloud_helpers && RUN_DATASET=<dataset> run_remote "diffusion/diffusion_train.py --mode diagnostic"`
   - `gan`: `source gcloud_helpers && RUN_DATASET=<dataset> run_remote "gan/model.py --train --refresh"`
   - `classifier`:
     - `gtex` → `source gcloud_helpers && run_remote "classifer/Classifier.py"` (single softmax)
     - `rnaseqdb` → `source gcloud_helpers && RUN_DATASET=rnaseqdb run_remote "classifer/MultiHeadClassifier.py --report-slices"`
       (one head per attribute; `--report-slices` prints the TCGA-only and normals-only
       confound controls, which are the numbers actually worth reading)
   - `classifier-small`: `source gcloud_helpers && RUN_DATASET=<dataset> run_remote "classifer/ClassiferSmall.py --split vinas"`
     ~464k params against Classifier.py's 122M, and dataset-aware: it reads
     `y_primary_disease_or_tissue.npy` for gtex and `y_tissue.npy` (15-way) for rnaseqdb.
     `--split vinas` reproduces the reference paper's procedure; add a second run with
     `--split donor` when you want the donor-leak-free number alongside it.
   - `fidelity`: the M3.5 encoding gate. CPU-bound and finishes in minutes, so run it in the
     foreground rather than detached:
     `source gcloud_helpers && RUN_DATASET=<dataset> run_remote_fg "evaluation/roundtrip_fidelity.py"`
     Skip steps 7-8 for this one — there are no training logs to tail; report the gate table
     directly. Results land in `output/evaluation/<dataset>/`.
   - `synthetic`: the M6 TSTR comparison against Viñas et al. §5.2.2. rnaseqdb only. This is
     a **sequence**, and every step depends on the one before it — do not start it unless a
     split-restricted diffusion checkpoint already exists, i.e. a run that was launched with
     `--split vinas`. A model trained without that flag has seen the test set and every number
     downstream of it is meaningless.

     1. Train (if not already done), detached, then wait for it:
        `RUN_DATASET=rnaseqdb run_remote "diffusion/diffusion_train.py --mode diagnostic --split vinas"`
     2. Pick the guidance scale on a subset — four short foreground runs, ~512 samples each:
        `RUN_DATASET=rnaseqdb run_remote_fg "evaluation/synthetic_fidelity.py --checkpoint <ema> --mode diagnostic --max-samples 512 --guidance-scale <w>"`
        for w in 1.0 2.0 3.0 5.0, and take the best gamma S_dist against the M3.5 ceiling.
     3. Generate the replica at the winning w, detached — this is hours, not minutes:
        `RUN_DATASET=rnaseqdb run_remote "diffusion/generate_synthetic_replica.py --checkpoint <ema> --mode diagnostic --guidance-scale <w>"`
        It writes a 7.2 GB array. **Never `/sync` it down**; everything that reads it runs on
        the VM. It resumes from `progress.json`, so a re-launch after an interruption is safe.
     4. Four classifier runs, each a few minutes, `--runs 5`:
        `RUN_DATASET=rnaseqdb run_remote_fg "classifer/ClassiferSmall.py --split vinas --attribute {tissue,condition} --runs 5 --out-suffix {real,synth}_{tissue,condition} [--synthetic-dir output/diffusion/diagnostic/rnaseqdb/synthetic_w<w>]"`
        The two real-trained runs omit `--synthetic-dir`; the two synthetic-trained runs pass it.
     5. Report: `RUN_DATASET=rnaseqdb run_remote_fg "evaluation/tstr_report.py"`, then relay the
        table. `/sync classifier` and `/sync eval` bring the JSONs down (small).

   Aside from the classifier's two variants, script paths do not change per dataset — only the
   env var does.

7. **Stream logs.** Read the `tail_logs` function body from `gcloud_helpers` and run the
   underlying `gcloud compute ssh ... sudo docker logs -f --tail 300 <container>` command as a
   background Bash task, then watch it via Monitor. Watch for "Training complete!" (success),
   an unhandled Python traceback, or the container exiting — whichever comes first. Tell the
   user up front that saying "stop logs" stops tailing without stopping the run.

8. **Report and hand off.** Once training finishes (or errors), summarize what happened and
   tell the user to run `/sync <module>` to pull results down locally. Do not run `/sync`
   automatically — the user may want to keep training or inspect the VM first.

9. **Offer to stop the VM.** Ask via AskUserQuestion whether to stop it now (options: stop it
   now / leave it running). If yes: `source gcloud_helpers && gcloud compute instances stop $INSTANCE_NAME --zone=$GCLOUD_INSTANCE_ZONE`.
   Never stop it without this explicit confirmation, even at the end of a run this same
   pipeline started.

Notes:
- Every VM start and every VM stop needs its own explicit confirmation in that run of the
  pipeline — a "yes" to one is not standing consent for the other, or for a future run.
- If the user interrupts between steps, don't resume later steps automatically on the next
  message unless they re-invoke `/pipeline` or explicitly say to continue.
