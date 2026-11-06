# AI release candidate — final local verification 2026-11-06

The Python service, backend and both clients share required review metadata and official wholesale provenance. Untrained, unfamiliar, unsupported and uncertain photos cannot become actionable grade recommendations. A matching trained checkpoint and photo reference are checked independently of health liveness. Production requires an internal token and disallows untrained preview mode.

Price snapshots validate source/date/unit/market/range structure before serving. Refresh writes validate first, fsync a temporary file and atomically replace the snapshot; failed refreshes preserve the prior data. The shipped snapshot remains dated 2026-10-22. No refresh or production request was performed. Interfaces disclose historical wholesale context and preserve manual pricing/grading.

Photo references reject malformed, nonfinite, unnormalized, mismatched-fingerprint or wrong-dimension embeddings. Release artifact validation records hashes and refuses missing trained weights/reference or confidence thresholds below 0.7.

Docker and CI consume the same 41 pinned Linux amd64/Python3.11 serving dependencies. Container smoke exposed an installed-wheel default-path error; explicit /app checkpoint/config/reference/price paths fix it. The completed image `sha256:b808f89e7b0e91a5eac1868d8249a921de18ce0838d780fe2a6441ceed12ca25` was retained from 2026-10-29. On 2026-11-06 eight installed service/runtime-lock/validator source hashes matched the workspace. A redundant rebuild was cancelled during base-image download; it did not replace this image. Base image tags remain mutable; this exact local image ID identifies the verified artifact.

Actual local container HTTP smoke passed with the frozen checkpoint mounted read-only and serving only 127.0.0.1:18000: trained model; photo check enabled; authenticated Wheat/kg wholesale response with source/date/market provenance; unauthenticated inference 401; synthetic blank JPEG referred as unfamiliar_photo. The artifact validator inside this same image confirms all four hashes match the prior local evidence. See ai-container-smoke.json, ai-container-artifacts.json and ai-image-source-hashes.txt. The smoke script uses an explicitly local fixture token, never a production credential.

The full source suite previously passed 61 tests and scoped Ruff passed. The fresh final suite also passed **61 tests** (44.74s); receipt is ai-tests.log. Curated tests, trained health and similarity checks do not establish field accuracy. Physical produce-photo evaluation remains a release acceptance gate; manual review must remain available.

## Develop integration — 2026-11-06

Merged the release work onto develop based on freshly fetched origin/main. Final merged-tree validation passed: 61 tests. Production was not contacted or deployed. Signing/device/provider acceptance gates remain as recorded above.
