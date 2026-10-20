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

## Using it from other code (CLI + REST API)

Start the server once (it owns the GPU, runs jobs one at a time, and applies the RAM cap):

    source ./env.sh && python ui/app.py            # http://127.0.0.1:7860
    PORT=8000 HOST=0.0.0.0 python ui/app.py        # other port / reachable on LAN (no auth – trusted network only)

Jobs submitted by scripts also show up in the web UI history. Results stay in output/ui_runs/<job id>/.

### Targets – "make every image 4K"

A target fits the output inside a standard frame **in the source's own orientation**, keeping the aspect ratio:
short side → target height, long side capped at target width.

| target          | short side | long side max | 16:9 result | 4:3 result | portrait 9:16 |
|-----------------|-----------:|--------------:|-------------|------------|---------------|
| `720p`          | 720        | 1280          | 1280×720    | 960×720    | 720×1280      |
| `1080p`         | 1080       | 1920          | 1920×1080   | 1440×1080  | 1080×1920     |
| `1440p` / `2k`  | 1440       | 2560          | 2560×1440   | 1920×1440  | 1440×2560     |
| `2160p` / `4k`  | 2160       | 3840          | 3840×2160   | 2880×2160  | 2160×3840     |

Images are never stretched or cropped; a 2:3 portrait photo at `4k` becomes 2160×3226 (short side 2160 –
the long-side cap only kicks in for wider-than-16:9 shapes). For a custom size pass `resolution` (short side, px)
and optionally `max_resolution` (long-side cap, 0 = none) instead of a target.
Big jumps (>~4×, e.g. a 320 px image → 4K) work but the model invents a lot of detail.
Rough speed here: a 4K image ≈ 40–45 s; video 1080p ≈ 5.5 s per frame.

### CLI – `seedvr2.py` (standard library only, any Python ≥ 3.10)

    python seedvr2.py photo.jpg                          # -> photo_4k.png next to the input
    python seedvr2.py input/images -o output/4k          # every image/video in a folder
    python seedvr2.py input/images -o output/4k --skip-existing
    python seedvr2.py clip.mp4 --target 1080p            # video -> clip_1080p.mp4 (audio kept)
    python seedvr2.py a.png --resolution 3000 --max-resolution 0
    python seedvr2.py a.png --model seedvr2_ema_7b_sharp_fp16.safetensors --seed 7

All files are queued at once, then downloaded as each finishes. Exit code 1 if any file failed.
`--server URL` or env `SEEDVR2_URL` points it at another machine.

From Python (copy `seedvr2.py` next to your code or add this folder to `sys.path`):

    from seedvr2 import upscale, submit, wait, download, SeedVR2Error
    out = upscale("photo.jpg", "photo_4k.png", target="4k")           # blocks, returns Path

    job = submit("photo.jpg", target="4k")                            # non-blocking
    wait(job["id"]); download(job["id"], "photo_4k.png")

### REST API

| method | path                          | what |
|--------|-------------------------------|------|
| POST   | `/api/v1/upscale`             | multipart upload; queues a job |
| GET    | `/api/v1/jobs/{id}`           | status: `queued` / `running` / `finalizing` / `done` / `error` / `stopped`, `percent`, `eta_s`, `width`, `height` |
| GET    | `/api/v1/jobs/{id}/result`    | the output file (PNG for images, MP4 for videos); 409 until done |
| GET    | `/api/v1/targets`             | list of target names |

`POST /api/v1/upscale` form fields: `file` (required), `target` (default `4k`), `resolution`, `max_resolution`,
`dit_model`, `seed` (42), `color_correction` (`lab`), `batch_size` (5, video only), `wait` (false), `timeout` (3600 s).

- `wait=false` → returns the job JSON at once; poll `status_url`, then fetch `result_url`.
- `wait=true`  → the response **is** the upscaled file (header `X-Job-Id`). 500 + job JSON if it failed,
  202 + job JSON if `timeout` ran out (keep polling).
- 400 for an unknown target, unsupported file type, or a job that would not fit in RAM/VRAM/disk.

    # one-shot: get the 4K file back directly
    curl -F file=@photo.jpg -F target=4k -F wait=true -o photo_4k.png http://127.0.0.1:7860/api/v1/upscale

    # async
    curl -F file=@photo.jpg -F target=4k http://127.0.0.1:7860/api/v1/upscale      # -> {"id": "...", ...}
    curl http://127.0.0.1:7860/api/v1/jobs/<id>
    curl -o photo_4k.png http://127.0.0.1:7860/api/v1/jobs/<id>/result

Don't run `src/.../inference_cli.py` by hand while the server is working – two jobs on the GPU at once run out of memory.
