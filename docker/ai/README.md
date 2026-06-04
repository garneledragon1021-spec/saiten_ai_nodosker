# Docker (AI service)

Build the image (run from `docker/ai`):

```bash
docker build -t saiten-ai:latest .
```

Run the container (default argument `test` can be changed):

# Linux / macOS / Git Bash / WSL
```bash
# runs: python3 all.py test
docker run --rm -v $(pwd)/../images:/images -v $(pwd)/../result:/result saiten-ai:latest

# or pass a different file name (without extension)
docker run --rm -v $(pwd)/../images:/images -v $(pwd)/../result:/result saiten-ai:latest your_image_name_without_ext
```

# PowerShell
```powershell
# runs: python3 all.py test
docker run --rm -v "${PWD}/../images:/images" -v "${PWD}/../result:/result" saiten-ai:latest

# or pass a different file name (without extension)
docker run --rm -v "${PWD}/../images:/images" -v "${PWD}/../result:/result" saiten-ai:latest your_image_name_without_ext
```

Notes:
- The container installs a CPU-only PyTorch build via the PyTorch CPU wheel index.
- `libzbar0` is installed in the container so `pyzbar` can load `zbar`.
- If the image build fails due to very large wheel downloads, consider building on a machine with more RAM or using a prebuilt base image that already contains `torch`.
- Use Bash-compatible shells (`bash`, Git Bash, or WSL) for the `$(pwd)` examples. In PowerShell, use `${PWD}` instead.
