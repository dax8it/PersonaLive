# PersonaLive offline inference on macOS Apple Silicon (MPS)

This branch is intentionally scoped to offline PyTorch MPS inference on Apple Silicon.

We are not pretending CUDA exists on Mac. CUDA-only paths stay CUDA-only:

- TensorRT is not supported on Mac. It requires NVIDIA CUDA, TensorRT, and pycuda.
- xformers is not supported on MPS. It is automatically disabled outside CUDA.
- Online/web streaming is deferred. First target is a verified offline render path.

## Hardware target

Target machine: MacBook Pro M2 with 96GB unified memory.

96GB should be enough memory for offline inference. The limiting factor is MPS compute throughput and unsupported CUDA-only acceleration, not RAM.

## Setup

Use Python 3.10 or 3.11. Avoid Python 3.13 for now because several ML/video packages may not ship wheels.

```bash
cd PersonaLive
python3.10 -m venv .venv-mps
source .venv-mps/bin/activate
python -m pip install -U pip wheel setuptools
pip install -r requirements_macos_mps.txt
```

Set MPS CPU fallback before running. The code also sets this by default, but exporting it before Python starts is safest:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Install/download weights exactly as the upstream README describes under `./pretrained_weights`.

## Verify MPS is visible

```bash
python - <<'PY'
import torch
print('torch', torch.__version__)
print('mps built', torch.backends.mps.is_built())
print('mps available', torch.backends.mps.is_available())
PY
```

## Offline smoke test

Start small first:

```bash
python inference_offline.py \
  --device mps \
  --config configs/prompts/personalive_offline_mps.yaml \
  --use_xformers false \
  --stream_gen true \
  -W 512 -H 512 -L 16
```

If that works, increase `-L`. If you hit an MPS kernel issue, retry with CPU fallback exported:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python inference_offline.py \
  --device mps \
  --config configs/prompts/personalive_offline_mps.yaml \
  --use_xformers false \
  -L 16
```

For performance experiments after correctness is proven, change `weight_dtype` in `personalive_offline_mps.yaml` from `fp32` to `fp16`.

## Online/web UI

Not supported by this branch yet.

The online path still follows the upstream CUDA/CPU behavior. Do not use it as evidence that MPS offline support is broken or working. After offline MPS generation is verified with real weights, online can be added as a separate branch with explicit MPS process/queue handling and latency tuning.

## Implementation notes

The port deliberately does not fake CUDA. It adds:

- `src/utils/device.py` with `resolve_device(auto|cuda|mps|cpu)` for offline inference
- safe CUDA/MPS cache helpers
- portable random generator creation for MPS
- automatic xformers disablement on non-CUDA devices
- MPS-specific offline config and requirements
- OpenCV video decoding fallback because decord has no macOS arm64 wheels

The core model is more MPS-friendly than it first appears: the so-called 3D convolutions are implemented as inflated 2D convolutions by reshaping frames into batch, which avoids many unsupported true-Conv3D MPS paths.
