# Anonymous review distribution

The full archival bundle contains six reference safetensors larger than the
8 MiB per-file limit used by some anonymous repository mirrors. Do not rely on
a GitHub Release URL to remain anonymous.

Build the separate review archive from a clean release tree:

```bash
python3 scripts/build_anonymous_review.py \
  --output ../LoRango_SDXL_Reproducibility_Bundle_v1.4.1_anonymous.tar.gz
```

The builder replaces every file larger than 7 MiB with deterministic numbered
chunks and adds `LARGE_ASSETS.json`. Each chunk and reconstructed file is bound
by SHA-256. Extract that archive into the repository uploaded to the anonymous
service.

After a reviewer downloads or clones the anonymous tree, restore the original
files and remove the tracked chunks from the local working tree:

```bash
python3 scripts/reassemble_large_assets.py --cleanup-parts
python3 scripts/submission_check.py
```

Reassembly refuses missing, reordered, modified, oversized, or hash-mismatched
chunks. After local reassembly the original files are untracked and the chunk
files appear deleted in the anonymous repository working tree; this is expected
and does not modify the remote review repository. The restored files are
verified by the normal release and reference manifests.

The anonymous review URL, not the source GitHub URL or a release-asset URL,
should be placed in double-blind submission material. Keep management tokens
private. Binary chunks are served unchanged, so the normal release anonymity
scan also checks binary bytes for forbidden machine-local identity markers.
