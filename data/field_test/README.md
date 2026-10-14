# Field-test photos (domain-shift reality check)

Drop REAL phone photos here, one folder per crop, as a farmer would actually take them
(whole crop / pile / bunch, phone camera, natural light, normal background — NOT isolated
lab grains). 10-30 per crop is plenty for a first read.

    data/field_test/wheat/   <- your wheat photos
    data/field_test/rice/
    data/field_test/mango/
    data/field_test/maize/

Then tell Claude, or run:
    PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/predict.py --dir data/field_test/wheat --crop wheat

## Labeled diagnostic evaluation

The 14 October 2026 collection contains 25 decodable images. Labels are read from the
crop directory and filename (`A.jpg`, `A1`, `B2.webp`, etc.). Extensionless images are
supported by the research evaluator; the older `predict.py` directory filter skips them.

From the AI repository, create a new immutable run:

```bash
./.venv/bin/python scripts/evaluate_field.py \
  --output ../research/results/field-diagnostic-NEW-RUN-ID
```

The evaluator snapshots the images, compares five frozen checkpoints, saves per-image
probabilities and metrics, and checks exact/pHash overlap against public representatives.
It does not train. Existing output directories are never overwritten.

The original collection is now an inspected development diagnostic. Preserve it and collect
fresh, sample/session/site-separated images for any future untouched adaptation test.
Record source, sample/lot, site, session, capture date/device, expert/farmer labeler IDs,
rubric version, and disagreements. Review visibly watermarked files individually.

Full results: `/Users/junaid/FYP/research/results-field-diagnostic-2026-10-14.md`.
