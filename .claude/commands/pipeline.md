Run the full remote workflow end-to-end: deploy the current code, get the GCP VM running,
pull the new image, and kick off a training run — pausing to confirm with the user at every
step that costs money or needs something only they can do.

The user may pass a module as $ARGUMENTS: `gan`, `diffusion` (default), or `classifier`. For
`diffusion`, training always uses `--mode diagnostic` (the only remote-capable diffusion
config — see `diffusion/diffusion_config.py`).

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

6. **Kick off training (detached).** Based on the module from $ARGUMENTS:
   - `diffusion` (default): `source gcloud_helpers && run_remote "diffusion/diffusion_train.py --mode diagnostic"`
   - `gan`: `source gcloud_helpers && run_remote "gan/model.py --train --refresh"`
   - `classifier`: `source gcloud_helpers && run_remote "classifer/Classifier.py"`

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
