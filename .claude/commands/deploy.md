Push the current code to GCP Artifact Registry.

The user may pass an optional flag as $ARGUMENTS:
- (no arg): use `push_gcloud_image_fast` — rebuilds only the runtime layer (code changes, ~1-2 min)
- `--full`: use `push_gcloud_image` — full rebuild including dependencies (~10-15 min, use when environment.yml or requirements.txt changed)
- `--base`: use `push_gcloud_base_image` — rebuild only the base/deps layer (rare)

Steps:
1. Run the appropriate build command based on $ARGUMENTS, prefixing with `source gcloud_helpers &&` to load helpers inline (e.g. `source gcloud_helpers && push_gcloud_image_fast`).
2. Once the push completes successfully, run `source gcloud_helpers && set_up_docker_image` to have the VM pull the latest image (re-auths to the registry, pulls the image, and mounts volumes).
3. Confirm the image is live and tell the user they can now run `/train <module>` or `run_remote "<script>"`.

If the build fails, show the last 50 lines of output and suggest common fixes (auth: `docker_login_remote`; dependency change: use `--full`).
