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

## Using it from other code (standalone, no server)

`seedvr2.py` runs the 7B FP16 model on the GPU directly – no UI or server needed. Each file runs in its own
process that exits when done, so RAM and VRAM are fully freed after every file. Calls from different scripts wait
for each other (one GPU job at a time, lock file `cache/gpu.lock`). The 16.5 GB model doesn't fit in the 16 GB card,
so it stays in RAM and is streamed to the GPU block by block; all compute is on the GPU.
A job needs ≈17.5 GB free RAM; if there isn't enough it fails at once with "not enough free RAM" (nothing starts).

    python seedvr2.py photo.jpg                          # -> photo_4k.png next to the input
    python seedvr2.py input/images -o output/4k          # every image/video in a folder
    python seedvr2.py input/images -o output/4k --skip-existing
    python seedvr2.py clip.mp4 --target 1080p            # video -> clip_1080p.mp4 (audio kept)
    python seedvr2.py a.png --resolution 3000 --max-resolution 0

From Python (add this folder to `sys.path`):

    from seedvr2 import upscale, SeedVR2Error
    out = upscale("photo.jpg", "photo_4k.png", target="4k")           # blocks, returns Path

Targets fit the output in a standard frame in the source's own orientation (short side → target height, long side
capped): 720p (720/1280), 1080p (1080/1920), 1440p|2k (1440/2560), 2160p|4k (2160/3840). A 4K image ≈ 20–25 s.

The old web UI (`ui/app.py`) is not needed anymore; don't run it at the same time as `seedvr2.py`.
