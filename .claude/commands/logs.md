Tail the training logs from the running container on the GCP VM.

The default container name is `ip3g`. If $ARGUMENTS is provided, use it as the container name (e.g., `ip3g-preprocess`).

Steps:
1. Source `gcloud_helpers` inline — do not check whether it is already sourced. Read the `tail_logs` function body from the `gcloud_helpers` file and run the underlying `gcloud compute ssh ... sudo docker logs -f --tail 300 <name>` command directly in a Bash background task, then stream it via Monitor.
2. The container name is `$ARGUMENTS` if provided, otherwise `ip3g`.
3. Tell the user Ctrl+C stops tailing but leaves the container running.

If the container isn't found, suggest running `gcloud compute ssh mark-test-instance --zone=us-central1-b --command="sudo docker ps"` to see what's running.
