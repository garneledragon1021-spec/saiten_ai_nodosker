# 1.2.0 accuracy evaluation protocol

Production APK: `dist/ScoreCamera-1.2.0-debug.apk`. Production inference code and
thresholds are frozen. Only an instrumentation benchmark and analysis tools are added.
No new model training, threshold tuning, or APK update is part of this evaluation.

## Ground truth and scope

- All 93 existing `digit_detector/images/val` crops were visually audited before inference.
  `manual_truth.csv` is the human-readable ground truth, not the YOLO pseudo-label files.
- These crops come from 20 existing source photos, not 93 independent writers.
  They belong to an earlier fine-tuning validation split; the original base model's
  training provenance is unknown. Do not call this a fresh independent holdout.
- The crop generator excluded photos/crops it could not auto-label (9 across the
  full original dataset). Consequently this is a selected subset, not an unbiased
  all-camera-input sample.
- Three sideways crops are explicitly rotated in the manifest preparation, based
  on visual inspection, before constructing rows. This is evaluation-data preparation,
  not an app capability. No image is altered to change its answer.
- Consecutive crops are arranged into 23 synthetic rows of four fields (last row five),
  retaining printed frame shape/aspect ratio and using the fixture's real QR.
- Six predeclared conditions: baseline; entire row resized to width 480;
  brightness multiplied by 0.45; horizontal illumination 0.30→1.0;
  Gaussian blur sigma 1.5px; JPEG quality 35. All variants reuse the same handwriting.
- 138 constructed cases, 558 field appearances, but only 93 distinct source fields.
- Existing sample sheet at longest sides 2400/1920/1280/1080/720: five further cases.
- Existing reported screenshot: automatic QR and injected QR evaluated separately.
  Never include injected QR as proof of end-to-end camera success.
- Total: 145 cases, 581 field appearances. They are correlated, not 145 independent photos.

## Metrics

- Correct field coverage = correct non-null fields / all expected fields.
- Accepted-field accuracy = correct non-null fields / all non-null fields.
- Wrong fields remain errors even when another unread field prevents a total.
- Complete-sheet exact match compares every field, not just the sum (compensating
  errors can produce a correct sum).
- Wrong complete sheets / wrong totals are separate safety indicators.
- QR/frame failures count as unread expected fields, not as correct outcomes.
- Time includes QR, image correction, frame detection, and inference on an Android 15
  x86_64 emulator. It is not Motorola performance, and excludes camera AF/capture/UI
  and the live mode's two-observation wait.

## Reproduction

1. `python tools/prepare_accuracy_evaluation.py` (Pillow, NumPy, OpenCV).
2. Build `assembleDebugAndroidTest` and install the **unchanged** 1.2.0 APK plus test APK.
3. Transfer `inputs/` to the target app's external files `accuracy-evaluation/` folder.
   Ensure the directory belongs to the app UID, not the ADB shell UID. For this local
   debuggable emulator only, ownership was corrected with root ADB. No phone root is needed
   for using the actual app; this is only benchmark fixture transport.
4. Run instrumentation class `jp.ebina.scorecamera.AccuracyEvaluationTest`.
5. Retrieve `results.jsonl` from that folder to this evaluation folder.
6. `python tools/summarize_accuracy_evaluation.py` produces `summary.json` and `cases.csv`.

Raw case inputs/results and the manual labels remain available for auditing.
The benchmark skips when its external fixtures have not been prepared.
