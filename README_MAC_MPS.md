# PersonaLive on macOS Apple Silicon (MPS)

This branch adds a first-class PyTorch MPS path for Apple Silicon Macs.

Status:

- Offline inference: intended to work on MPS after weights are installed.
- Online/web UI: intended to start on MPS with `--acceleration none`, but expect lower FPS than CUDA/TensorRT.
- TensorRT: not supported on Mac. It requires NVIDIA CUDA, TensorRT, and pycuda.
- xformers: not supported on MPS. It is automatically disabled outside CUDA.

## Hardware target

Test target requested: MacBook Pro M2 with 96GB unified memory.

96GB is enough memory for the model. The limitation is MPS compute throughput and missing CUDA-only acceleration, not RAM.

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

Run without CUDA-only acceleration:

```bash
python inference_online.py \
  --device mps \
  --acceleration none \
  --config_path ./configs/prompts/personalive_online_mps.yaml
```

Open:

```text
http://localhost:7860
```

If latency is too high, reduce workload before changing model code:

- lower `height`/`width` in `configs/prompts/personalive_online_mps.yaml`
- keep `num_inference_steps: 4`
- keep `temporal_window_size: 4`
- keep `--acceleration none`

## Implementation notes

The port deliberately does not fake CUDA. It adds:

- `src/utils/device.py` with `resolve_device(auto|cuda|mps|cpu)`
- safe CUDA/MPS cache helpers
- portable random generator creation for MPS
- automatic xformers disablement on non-CUDA devices
- online `--device` CLI support
- MPS-specific configs and requirements
- OpenCV video decoding fallback because decord has no macOS arm64 wheels
- CPU tensor queues between the web process and generation process, then device transfer inside the child process

The core model is more MPS-friendly than it first appears: the so-called 3D convolutions are implemented as inflated 2D convolutions by reshaping frames into batch, which avoids many unsupported true-Conv3D MPS paths.
