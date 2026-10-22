#!/usr/bin/env python3
"""SeedVR2 7B FP16 upscaler – runs the model on the GPU directly, no server or UI needed. Standard library only.

  python seedvr2.py photo.jpg                      # -> photo_4k.png next to it
  python seedvr2.py input/images -o output/4k      # every image in the folder
  python seedvr2.py clip.mp4 --target 1080p
  python seedvr2.py a.png --resolution 3000 --max-resolution 0

From Python:  from seedvr2 import upscale;  upscale("photo.jpg", "photo_4k.png", target="4k")

Every call starts the model in its own process and exits when done, so RAM and VRAM are fully freed after each
file. Calls from different scripts wait for each other (one GPU job at a time). The 16.5 GB FP16 model is larger
than the 16 GB card, so it is kept in RAM and streamed to the GPU block by block (block swap); all compute is on
the GPU. A job needs ≈17.5 GB free RAM; it runs under a hard RAM cap so a shortfall kills the job, not the desktop.
"""
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CLI = ROOT / "src/SeedVR2_VideoUpscaler/inference_cli.py"
PYTHON = ROOT / "venv/bin/python"
MODEL_DIR = ROOT / "models/SEEDVR2"
TMP = ROOT / "cache/tmp"
LOCK = ROOT / "cache/gpu.lock"
DEFAULT_MODEL = "seedvr2_ema_7b_fp16.safetensors"

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
MEDIA_EXT = IMAGE_EXT | VIDEO_EXT
TARGETS = {"720p": (720, 1280), "1080p": (1080, 1920), "1440p": (1440, 2560), "2k": (1440, 2560),
           "2160p": (2160, 3840), "4k": (2160, 3840)}

GB = 1024 ** 3
RAM_HEADROOM = 4 * GB     # a job never gets more than MemTotal minus this
FREE_RAM_RESERVE = GB     # free RAM left outside a job's cap; MemAvailable already excludes what the desktop holds
RUNTIME_RAM = int(1.5 * GB)  # python + torch + CUDA context, on top of the weights
VAE_RAM = GB // 2
BYTES_PER_OUT_PX = 60     # video: RAM per output pixel per chunk frame (measured ≈45 at 1080p + margin)
MAX_CHUNK = 60


class SeedVR2Error(RuntimeError):
    pass


def mem_info() -> dict:
    vals = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":")
        vals[k] = int(v.split()[0]) * 1024
    return vals


def ram_limit() -> int:
    """Hard RAM cap for the job: what is free now minus a small reserve for the desktop."""
    m = mem_info()
    return min(m["MemTotal"] - RAM_HEADROOM, m["MemAvailable"] - FREE_RAM_RESERVE)


def probe(path: Path) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                        "stream=width,height,nb_read_packets", "-of", "json", str(path)],
                       capture_output=True, text=True)
    try:
        s = json.loads(r.stdout)["streams"][0]
        return {"width": int(s["width"]), "height": int(s["height"]), "frames": int(s.get("nb_read_packets") or 1)}
    except (ValueError, KeyError, IndexError):
        raise SeedVR2Error(f"cannot read {path} (ffprobe: {r.stderr.strip()[:200]})") from None


def even(x: float) -> int:
    return max(2, int(round(x / 2)) * 2)


def output_size(w: int, h: int, resolution: int, max_resolution: int) -> tuple[int, int]:
    """Mirror of the CLI's sizing: short side -> resolution, long side capped by max_resolution."""
    s = resolution / min(w, h)
    if max_resolution > 0 and max(w, h) * s > max_resolution:
        s = max_resolution / max(w, h)
    return even(w * s), even(h * s)


def resolve_size(target: str, resolution: int, max_resolution: int) -> tuple[int, int]:
    if resolution > 0:
        return resolution, max(0, max_resolution)
    if target.lower() not in TARGETS:
        raise SeedVR2Error(f"unknown target {target!r}; use one of {sorted(TARGETS)} or pass resolution")
    res, cap = TARGETS[target.lower()]
    return res, cap if max_resolution < 0 else max_resolution


def add_audio(src: Path, video: Path):
    """The CLI writes video only; copy the source audio track back in."""
    has_audio = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                                "-of", "csv=p=0", str(src)], capture_output=True, text=True).stdout.strip()
    if not has_audio:
        return
    tmp = video.with_name("with_audio.mp4")
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-i", str(src), "-map", "0:v",
                        "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(tmp)])
    if r.returncode == 0 and tmp.exists():
        tmp.replace(video)


def upscale(path: str | Path, out: str | Path | None = None, target: str = "4k", resolution: int = 0,
            max_resolution: int = -1, dit_model: str = DEFAULT_MODEL, seed: int = 42,
            color_correction: str = "lab", batch_size: int = 5, verbose: bool = False, **_ignored) -> Path:
    """Upscale one image or video and return the output path. Blocks while another job holds the GPU.
    Raises SeedVR2Error if it fails or there is not enough free RAM right now."""
    src = Path(path).resolve()
    if not src.is_file():
        raise SeedVR2Error(f"not found: {src}")
    if src.suffix.lower() not in MEDIA_EXT:
        raise SeedVR2Error(f"unsupported file type: {src.suffix}")
    is_video = src.suffix.lower() in VIDEO_EXT
    res, cap = resolve_size(target, resolution, max_resolution)
    label = f"{resolution}p" if resolution > 0 else target.lower()
    out = Path(out) if out else default_output(src, None, label)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = probe(src)
    out_w, out_h = output_size(meta["width"], meta["height"], res, cap)

    TMP.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # one GPU job at a time, across all scripts
        dit = MODEL_DIR / dit_model
        model_ram = (dit.stat().st_size if dit.exists() else int(16.5 * GB)) + VAE_RAM + RUNTIME_RAM
        limit = ram_limit()
        per_frame = out_w * out_h * BYTES_PER_OUT_PX
        b = min(batch_size, meta["frames"]) if is_video else 1
        chunk = 1
        if is_video:
            chunk = max(b, min(MAX_CHUNK, int((limit - model_ram) // per_frame)) // b * b)
        need = model_ram + chunk * per_frame
        if need > limit:
            raise SeedVR2Error(f"not enough free RAM: needs ≈{need / GB:.1f} GB, the job can get {limit / GB:.1f} GB "
                               f"right now. Close other apps (browser tabs) and retry.")

        work = Path(tempfile.mkdtemp(prefix="seedvr2-", dir=TMP))
        try:
            tmp_out = work / ("output.mp4" if is_video else "output.png")
            cmd = [str(PYTHON), str(CLI), str(src), "--output", str(tmp_out),
                   "--output_format", "mp4" if is_video else "png",
                   "--model_dir", str(MODEL_DIR), "--dit_model", dit_model,
                   "--resolution", str(res), "--max_resolution", str(cap), "--batch_size", str(b),
                   "--seed", str(seed), "--color_correction", color_correction,
                   "--blocks_to_swap", "36", "--dit_offload_device", "cpu", "--vae_offload_device", "cpu",
                   "--vae_encode_tiled", "--vae_decode_tiled"]
            if is_video:
                cmd += ["--chunk_size", str(chunk), "--temporal_overlap", str(max(1, b // 4) if b >= 5 else 0)]
            if shutil.which("systemd-run"):
                cmd = ["systemd-run", "--user", "--scope", "--quiet", "-p", f"MemoryMax={limit}",
                       "-p", "MemorySwapMax=0", *cmd]
            env = {**os.environ, "TMPDIR": str(work), "HF_HOME": str(ROOT / "cache/huggingface"),
                   "TORCH_HOME": str(ROOT / "cache/torch"), "XDG_CACHE_HOME": str(ROOT / "cache")}
            log = work / "log.txt"
            with log.open("w") as lf:
                rc = subprocess.run(cmd, cwd=CLI.parent, env=env, stdout=None if verbose else lf,
                                    stderr=subprocess.STDOUT).returncode
            if rc != 0 or not tmp_out.exists():
                tail = "" if verbose else "\n" + "\n".join(log.read_text(errors="replace").splitlines()[-15:])
                why = (f"killed: ran out of its {limit / GB:.1f} GB RAM cap" if rc in (-9, 137)
                       else f"SeedVR2 exited with code {rc}")
                raise SeedVR2Error(why + tail)
            if is_video:
                add_audio(src, tmp_out)
            shutil.move(str(tmp_out), out)
        finally:
            shutil.rmtree(work, ignore_errors=True)  # temp frames/logs; RAM/VRAM are freed when the process exits
    return out


def default_output(src: Path, out_dir: Path | None, label: str) -> Path:
    suffix = ".mp4" if src.suffix.lower() in VIDEO_EXT else ".png"
    return (out_dir or src.parent) / f"{src.stem}_{label}{suffix}"


def collect(inputs: list[str]) -> list[Path]:
    files = []
    for i in map(Path, inputs):
        if i.is_dir():
            files += sorted(p for p in i.iterdir() if p.suffix.lower() in MEDIA_EXT and p.is_file())
        elif i.is_file():
            files.append(i)
        else:
            sys.exit(f"not found: {i}")
    return files


def main():
    ap = argparse.ArgumentParser(description="Upscale / restore images and videos with SeedVR2 7B on the local GPU.")
    ap.add_argument("inputs", nargs="+", help="image/video files or folders")
    ap.add_argument("-o", "--out-dir", type=Path, help="output folder (default: next to each input)")
    ap.add_argument("-t", "--target", default="4k", help="720p, 1080p, 1440p/2k, 2160p/4k (default 4k)")
    ap.add_argument("--resolution", type=int, default=0, help="exact short side in px (overrides --target)")
    ap.add_argument("--max-resolution", type=int, default=-1, help="long-side cap in px, 0 = none")
    ap.add_argument("--model", dest="dit_model", default=DEFAULT_MODEL, help=f"DiT weights (default {DEFAULT_MODEL})")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--color-correction", dest="color_correction", default="lab")
    ap.add_argument("--skip-existing", action="store_true", help="skip inputs whose output file already exists")
    ap.add_argument("-v", "--verbose", action="store_true", help="show the model's own output")
    a = ap.parse_args()

    label = f"{a.resolution}p" if a.resolution > 0 else a.target.lower()
    failed = 0
    for src in collect(a.inputs):
        out = default_output(src, a.out_dir, label)
        if a.skip_existing and out.exists():
            print(f"skip  {src} (exists: {out})")
            continue
        t = time.time()
        print(f"run   {src}", flush=True)
        try:
            upscale(src, out, a.target, a.resolution, a.max_resolution, a.dit_model, a.seed, a.color_correction,
                    verbose=a.verbose)
            print(f"done  {src} -> {out} ({time.time() - t:.0f} s)", flush=True)
        except SeedVR2Error as e:
            failed += 1
            print(f"FAIL  {src}: {e}", file=sys.stderr)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
