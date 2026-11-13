# Original digit detector used for conditional recovery in 1.3.0

Asset: `app/src/main/assets/score_detector_manual_v1.onnx`.

- Source repository: https://github.com/J2523-Sou/saiten_ai_ebina
- Checkpoint: `docker/ai/train_data/ref/best_detecter.pt` at pre-iOS commit `c99d471`.
- Git blob: `b3fe11ddfcd828ed2853bf1a78751d716f9a1402`.
- This is identical to the initial commit `e77b0d0` checkpoint at
  `docker/ai/program/detect/ref/best.pt`.
- Checkpoint SHA-256: `690a55bfde7fc88bbad93de28fe2a09dd36afeee833f02bacad52eb09b436cd2`.
- The user identified this as the model trained with manually annotated data.
  Training annotations themselves were not independently audited in this task.
- Stored training settings: YOLO11n, ten digit classes zero through nine,
  image size 512, configured epochs 100.
- Export: Ultralytics 8.4.60, ONNX opset 17, fixed input 1x3x640x640,
  no embedded NMS, no new training. See `tools/audit_historical_models.py`.

The original and auto-finetuned detector each read 30/32 manually located fields
correctly on the supplied eight photos, with different errors. This is not an
end-to-end accuracy claim and is not an independent holdout after development.

The auto-finetuned model remains the primary detector. This original model loads
lazily only if normal reading and a tighter crop leave a field uncertain. Its result
must agree with the foreground classifier or OCR and must not drop a previously
observed digit. It is never an unconditional override or a lookup of known answers.
