# Archival release procedure

The repository is prepared for an immutable archival upload but cannot create
an external DOI without the authors' repository/archive account.

Before submission:

1. run `bash scripts/artifact_check.sh`;
2. verify the preserved local v1.4 GPU evidence hashes and retain the original
   sealed evidence outside the anonymous bundle;
3. create the release archive with `python scripts/build_release.py`;
4. upload that exact archive to Zenodo, OSF, or an equivalent long-term public
   repository;
5. replace `IDENTIFIER_PENDING` in `CITATION.cff` and add the assigned
   DOI/identifier to the release metadata;
6. publish the archive URL and SHA-256 in the paper and artifact appendix.

For double-blind services with a per-file size limit, create the separate
split-asset archive described in `docs/ANONYMOUS_REVIEW.md`. It is a transport
variant of the same release tree, not the immutable archival object used for a
DOI.

Do not claim ACM **Available** before step 5 is complete. Functional,
Reproduced, and Reusable evidence is independent of the DOI but must point to
the exact same archived bytes.
