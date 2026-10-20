# SeedVR2 (7B, FP16) – image & video quality enhancement trial

Model: ByteDance-Seed/SeedVR2-7B (one-step diffusion restoration / upscaling)
Weights: https://huggingface.co/ByteDance-Seed/SeedVR2-7B
Easiest runner: ComfyUI-SeedVR2_VideoUpscaler (numz) – supports block swap for 16 GB GPUs

Layout:
  models/         model weights (seedvr2_ema_7b_fp16.safetensors ~16 GB + VAE)
  input/images/   test images
  input/videos/   test videos
  output/         enhanced results

Notes for this machine (RTX 4070 Ti SUPER 16 GB, 31 GB RAM):
  - 7B FP16 weights ≈ VRAM size → needs block swap / CPU offload, or use FP8 variant
  - Disk: ~25 GB free on / – check space before downloading weights

## Self-contained rule
EVERYTHING for this trial lives inside this folder so deleting it reclaims all space:
  - Python venv:   ./venv
  - Code/repos:    ./src
  - Caches:        ./cache (HF_HOME, TORCH_HOME, PIP_CACHE_DIR, XDG_CACHE_HOME)
Always `source ./env.sh` before installing or running anything.
Cleanup: rm -rf ~/Desktop/codes/SeedVR2-7B-FP16
