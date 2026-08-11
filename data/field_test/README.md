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
