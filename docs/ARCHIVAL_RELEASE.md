# Archival release procedure

The repository is prepared for an immutable archival upload but cannot create
an external DOI without the authors' repository/archive account.

Before submission:

1. run `bash scripts/artifact_check.sh`;
2. run and preserve the full reference GPU reproduction;
3. create the release archive with `python scripts/build_release.py`;
4. upload that exact archive to Zenodo, OSF, or an equivalent long-term public
   repository;
5. replace `IDENTIFIER_PENDING` in `CITATION.cff` and add the assigned
   DOI/identifier to the release metadata;
6. publish the archive URL and SHA-256 in the paper and artifact appendix.

Do not claim ACM **Available** before step 5 is complete. Functional,
Reproduced, and Reusable evidence is independent of the DOI but must point to
the exact same archived bytes.
