# Handwritten digit fallback

`app/src/main/assets/handwritten_digits_v1.onnx` is a small 28x28, ten-class CNN
trained locally by `tools/train_handwritten_digits.py` on the public MNIST dataset.

- Dataset: MNIST by Yann LeCun, Corinna Cortes and Christopher J.C. Burges.
- Dataset description: http://yann.lecun.com/exdb/mnist/
- Loader and verified download mirror: torchvision.datasets.MNIST
  (https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html).
- Training data: 60,000 MNIST training images; 4 epochs with random affine augmentation.
- Evaluation: the separate 10,000-image MNIST test split, evaluated after training.
- Recorded held-out accuracy: 0.9891. This is **not** score-sheet accuracy or
  performance measured on the Motorola device.
- Seed, input normalization and evaluation counts are in `handwritten_model_metrics.json`.
- User screenshots are used only for regression evaluation, never for training this model.

The application extracts visible glyphs from three foreground masks and requires
at least two to corroborate the largest observed count; a majority cannot vote
away an extra digit. Detached strokes with overlapping horizontal ranges are joined.
A field with uncertain count, conflicting complete readings or
unrecognized glyphs is left for review rather than silently dropping a digit.
The CNN complements the existing YOLO and bundled ML Kit recognizers; it does not
provide a guarantee that a confidently classified handwritten digit is correct.
