# Deadline MVP grading release

## Product contract

The farmer deliberately chooses wheat, rice, mango, maize or **Other crop**. Supported crops
can request an optional photo-based suggestion. Other makes no quality request, shows no model
grade, and retains manual listing. The backend rejects unsupported direct inputs before fetching
the AI service or storing a prediction; the AI service rejects them before inference.

Suggestions are applied only through farmer confirmation. Changing crop/photo/product invalidates
old results, late requests cannot restore them, and a previously applied suggestion is reverted to
the earlier manual grade unless the farmer already overrode it. D has no automatic listing-grade
mapping. English and Urdu copy identifies model output as a suggestion.

## Selected artifact

Freeze the previously established plain grade-only EfficientNet-B0 seed-42 best checkpoint
(epoch 15). There is no new training or field-set tuning in this release. The crop remains an
explicit supported-input gate and traceability input; the image model does not predict crop ID.

The package generated on 14 October 2026 is:

```text
models/exports/mvp-2026-10-14/best_model.pth
models/exports/mvp-2026-10-14/manifest.json
models/exports/mvp-2026-10-14/SHA256SUMS
```

Version: `grade-only-efficientnet_b0-492aa1ba`.
Artifact SHA-256:
`492aa1baa4ce5a32e0f09e92b7caddb24717e1f294768f6f745097d5f2d26b31`.
Size: 21,616,457 bytes. Only model state is exported; optimizer state is removed. Tensor equality
against the source checkpoint was verified. A fresh serialization may have a different file hash;
use the manifest produced for that export, not a hardcoded historical hash.

To regenerate in a new directory:

```bash
./.venv/bin/python scripts/export_mvp_checkpoint.py --output-dir models/exports/NEW-RELEASE
```

The artifact's local HTTP verification is recorded under
`/Users/junaid/FYP/research/results/mvp-grading-2026-10-14-01/`.

## EC2 rollout order

The existing protected-main CI/release agent remains the application deployment path. Changes
are currently local; neither application releases nor these weights have been activated on EC2.

1. Publish the tested backend and AI changes through the existing PR/main release flow. Deploy the
   matching web release and build the mobile release. Verify exact-SHA activation receipts and
   `sudo f2fctl check` as described in [ec2-deployment.md](ec2-deployment.md).
2. **Activate the new AI service code before replacing the old weights.** It reads both old
   conditioned and new plain weights. The previous service code cannot read the new artifact.
3. Transfer the three artifact files to a staging directory on EC2 using the existing SSH access.
   Verify `sha256sum -c SHA256SUMS` in that directory. Keep trained weights out of Git and images.
4. Retain the existing `/opt/farm2fork/models/best_model.pth` as a timestamped backup. Install the
   verified new file to a temporary name on the same filesystem, readable by container UID 10001,
   then atomically rename it to `best_model.pth`. Do not modify Fabric identities, ledger or volumes.
5. Recreate only AI using the authoritative Compose files:

```bash
sudo docker compose \
  --env-file /opt/farm2fork/backend.env \
  --env-file /opt/farm2fork/deployment-images.env \
  -f /opt/farm2fork/ops/compose.yml \
  up -d --no-deps --force-recreate ai
sudo f2fctl check
```

6. Confirm internal `/health` reports `trained`, all four supported crops and the manifest's exact
   `grade-only-...` version. The existing Compose mount has a historical `grade_cond_dedup` path;
   its explicit `MODEL_CHECKPOINT` override is still valid for new weights and need not be renamed.
   `ALLOW_UNTRAINED_MODEL=false` must remain set. `AI_REQUIRE_TRAINED=true` makes the existing
   readiness probe require trained weights.
7. Exercise authenticated farmer flows: one supported crop with a photo and explicit application,
   Other with manual grade, and supported→Other during an in-flight request. Other must generate
   no `/ai/quality` request. Check that backend direct `other`, cotton and sugarcane requests fail
   without a stored quality prediction. New AI direct requests return 422 without a grade.

For a weight rollback, atomically restore the retained previous weights and recreate AI. The new
service can load that legacy artifact. If application code is also reverted, restore compatible
conditioned weights **before** restoring old AI code. Do not reset the blockchain to roll back AI.

## Verification and limits

- AI suite: 38 passing tests, covering plain/legacy loading, supported-input gate, missing weights,
  mismatch handling, and absence of random default production grades.
- Real exported artifact: all 25 supported-photo predictions match the frozen baseline via HTTP;
  Other/cotton/sugarcane/banana each return 422 with zero grader calls.
- Backend: 19 focused tests and build passed. Web: 24 focused tests, TypeScript and ESLint passed.
- Mobile: 17 focused tests, bilingual listing checks and scoped analyzer passed; four pre-existing
  Firebase analyzer warnings remain.
- A local Docker image build was not verified because the Docker daemon is unavailable. Web
  production build was blocked by sandbox port permissions and Google Fonts DNS. Existing CI
  remains necessary before release. EC2, browser/device and authenticated end-to-end activation
  have not been performed for this change.

The model's seed-42 curated test accuracy is 439/466 (94.2%); its supplied-photo diagnostic agreement
is 7/25 (28%). Those figures concern different distributions. MVP flow completion does not establish
better field accuracy. Preserve the diagnostic and research limitations when presenting this release.
