"""Minimal test UI for SeedVR2: upload image/video, run inference_cli.py, compare input vs output.

Run:  source ./env.sh && python ui/app.py   ->  http://127.0.0.1:7860
"""
import json
import math
import re
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
CLI = ROOT / "src" / "SeedVR2_VideoUpscaler" / "inference_cli.py"
MODEL_DIR = ROOT / "models"
RUNS_DIR = ROOT / "output" / "ui_runs"
UPLOADS_DIR = RUNS_DIR / "_uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm"}

# Standard tiers, named by the short side (1080p = short side 1080 px, also for portrait)
STANDARD_TIERS = [(240, "240p"), (360, "360p SD"), (480, "480p SD"), (720, "720p HD"),
                  (1080, "1080p Full HD"), (1440, "1440p QHD / 2K"), (2160, "2160p 4K UHD")]

app = FastAPI()

# ---------- RAM safety ----------
# The 7B DiT is offloaded to system RAM (block swap), so RAM, not VRAM, is the limit for video.
# Without chunking the CLI keeps every decoded frame in RAM and froze the whole machine on a 10 s clip.
GB = 1024 ** 3
BYTES_PER_OUT_PX = 60     # measured ≈45 B per output pixel per chunk frame at 1080p (+ margin): decoded, assembled, colour-fixed copies
MAX_CHUNK = 60            # longer chunks barely speed things up, raise the RAM peak and lose more on a crash
RAM_HEADROOM = 4 * GB     # left for the desktop / browser


def mem_info() -> dict:
    vals = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":")
        vals[k] = int(v.split()[0]) * 1024
    return vals


def model_ram_bytes(dit_model: str) -> int:
    """RAM the offloaded DiT + VAE occupy while cached between chunks."""
    d = MODEL_DIR / "SEEDVR2"
    dit = d / dit_model
    size = dit.stat().st_size if dit.exists() else 16 * GB
    vae = d / "ema_vae_fp16.safetensors"
    return size + (vae.stat().st_size if vae.exists() else GB // 2)


def usable_ram() -> int:
    """Available RAM, counting memory held by any running SeedVR2 CLI (it is gone before the next job starts)."""
    held = 0
    for d in Path("/proc").iterdir():
        try:
            if d.name.isdigit() and b"inference_cli.py" in (d / "cmdline").read_bytes():
                held += int(next(l for l in (d / "status").read_text().splitlines() if l.startswith("VmRSS")).split()[1]) * 1024
        except (OSError, StopIteration, ValueError):
            pass
    return mem_info()["MemAvailable"] + held


def auto_chunk(out_w: int, out_h: int, batch_size: int, dit_model: str) -> int:
    """Largest chunk (multiple of batch_size) whose frames fit in available RAM."""
    budget = usable_ram() - model_ram_bytes(dit_model) - RAM_HEADROOM
    per_frame = out_w * out_h * BYTES_PER_OUT_PX
    frames = max(batch_size, min(MAX_CHUNK, int(budget // per_frame)))
    return max(batch_size, frames // batch_size * batch_size)


def output_size(w: int, h: int, resolution: int, max_resolution: int) -> tuple[int, int]:
    """Mirror of the CLI's sizing: short side -> resolution, long side capped by max_resolution."""
    s = resolution / min(w, h)
    if max_resolution > 0 and max(w, h) * s > max_resolution:
        s = max_resolution / max(w, h)
    return even(w * s), even(h * s)


def memory_capped(cmd: list[str]) -> tuple[list[str], int]:
    """Run inside a systemd scope with a hard RAM limit so an overrun kills only this job,
    instead of driving the desktop into swap until it freezes."""
    limit = mem_info()["MemTotal"] - RAM_HEADROOM
    if shutil.which("systemd-run"):
        return ["systemd-run", "--user", "--scope", "--quiet", "-p", f"MemoryMax={limit}",
                "-p", "MemorySwapMax=1G", *cmd], limit
    return cmd, 0


# ---------- time / resource estimates ----------
# Calibrated on this machine (RTX 4070 Ti SUPER 16 GB, 7B FP16, blocks_to_swap 36, tiled VAE):
#   1080p video, batch 5: encode 2.8 s + DiT 10 s + decode 6.3 s per batch; 4K image ≈ 40 s incl. load, 6.9 GB VRAM.
# Finished jobs refine these numbers through calibration_factor().
LOAD_S = 25               # python start + model load (warm page cache)
DIT_S_PER_LATENT_MP = 2.4 # DiT seconds per latent frame per output megapixel (7B)
VAE_S_PER_FRAME_MP = 0.87 # VAE encode + decode seconds per frame per output megapixel
CHUNK_OVERHEAD_S = 3      # cleanup + PNG writing between chunks
VRAM_BASE_GB = 1.9        # CUDA context + resident DiT parts with full block swap
VRAM_GB_PER_LATENT_MP = 0.6
VRAM_VAE_GB = 4.0         # tiled VAE peak
OUT_BITS_PER_PX = 0.35    # x264 crf 14 output, bits per pixel per frame
PNG_BYTES_PER_PX = 1.6    # temporary PNG frames waiting to be packed into segments


def dit_info(name: str) -> tuple[int, float]:
    """(RAM bytes the DiT occupies, compute factor relative to 7B)."""
    n = name.lower()
    speed = 0.45 if "3b" in n else 1.0
    f = MODEL_DIR / "SEEDVR2" / name
    if f.exists():
        return f.stat().st_size, speed
    gb = {("7b", "fp16"): 16.5, ("7b", "fp8"): 8.3, ("3b", "fp16"): 6.8, ("3b", "fp8"): 3.4}
    size = next((v for (m, q), v in gb.items() if m in n and q in n), None)
    if size is None and "gguf" in n:
        size = (8.0 if "q8" in n else 4.6) * (0.45 if "3b" in n else 1.0)
    return int((size or 16.5) * GB), speed


def calibration_factor(kind: str) -> float:
    """Median actual/estimated time of recent finished jobs of this kind (1.0 until there is history)."""
    ratios = sorted(j["stats"]["elapsed_s"] / j["estimate_s"] for j in list(jobs.values())[-40:]
                    if j.get("status") == "done" and j.get("type") == kind and j.get("estimate_s")
                    and (j.get("stats") or {}).get("elapsed_s") and not j.get("resumed"))
    return ratios[len(ratios) // 2] if ratios else 1.0


def fmt_s(s: float) -> str:
    s = int(s)
    return f"{s // 3600} h {s % 3600 // 60} min" if s >= 3600 else f"{s // 60} min {s % 60} s" if s >= 60 else f"{s} s"


def estimate(meta: dict, is_video: bool, p: dict) -> dict:
    """Time, RAM, VRAM and disk needed for one run, plus the limitations that apply to it."""
    w, h = meta.get("width") or 1920, meta.get("height") or 1080
    out_w, out_h = output_size(w, h, p["resolution"], p.get("max_resolution", 0))
    mp = out_w * out_h / 1e6
    frames = 1
    if is_video:
        frames = meta.get("frames") or int((meta.get("duration_s") or 0) * (meta.get("fps") or 25)) or 1
        if p.get("load_cap"):
            frames = min(frames, p["load_cap"])
    b = p.get("batch_size", 5) if is_video else 1
    b = min(b, max(1, frames)) if is_video else 1
    dit_bytes, speed = dit_info(p.get("dit_model", "seedvr2_ema_7b_fp16.safetensors"))
    swap = p.get("blocks_to_swap", 36)
    mem = mem_info()

    # RAM and chunking
    model_ram = dit_bytes + GB // 2 + int(1.5 * GB)   # DiT + VAE + python/torch
    per_frame = out_w * out_h * BYTES_PER_OUT_PX
    chunk = p.get("chunk_size") or 0
    if is_video and chunk <= 0:
        chunk = auto_chunk(out_w, out_h, b, p.get("dit_model", ""))
    chunk = min(chunk, frames) if is_video else 1
    ram_peak = model_ram + chunk * per_frame
    overlap = 0 if b < 5 else max(1, b // 4)

    # time
    latent = (b - 1) // 4 + 1
    step = max(1, b - overlap)
    batches = max(1, math.ceil(max(0, frames - overlap) / step)) if is_video else 1
    chunks = math.ceil(frames / chunk) if is_video else 1
    per_batch = (DIT_S_PER_LATENT_MP * latent * mp * speed) + VAE_S_PER_FRAME_MP * b * mp
    cold = mem.get("Cached", 0) < dit_bytes  # weights not in page cache -> read from disk
    load = LOAD_S + (dit_bytes / (1.5 * GB) if cold else 0)
    t = (load + batches * per_batch + chunks * CHUNK_OVERHEAD_S) * calibration_factor("video" if is_video else "image")

    # VRAM
    resident = dit_bytes * max(0, 36 - swap) / 36 / GB
    vram = max(VRAM_BASE_GB + resident + VRAM_GB_PER_LATENT_MP * latent * mp,
               VRAM_BASE_GB + resident + (VRAM_VAE_GB if p.get("vae_tiled", True) else 2.5 * mp * b))
    vram_total = 16.0
    try:
        vram_total = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                                        capture_output=True, text=True, timeout=5).stdout.split()[0]) / 1024
    except Exception:  # noqa: BLE001
        pass

    # disk
    out_bytes = out_w * out_h * frames * (OUT_BITS_PER_PX / 8 if is_video else 1.3)
    tmp_bytes = (out_bytes + (SEGMENT_FRAMES + chunk) * out_w * out_h * PNG_BYTES_PER_PX) if is_video else 0
    disk_free = shutil.disk_usage(RUNS_DIR).free

    limits, blockers = [], []
    scale = p["resolution"] / min(w, h)
    if scale < 0.95:
        limits.append("Target is smaller than the source – this downscales, it doesn't add detail.")
    elif scale > 4:
        limits.append(f"{scale:.1f}× jump: the model must invent most fine detail (faces, text, textures can drift "
                      "from the original). Two passes of ≤2.5× look more natural.")
    elif scale > 2.5:
        limits.append(f"{scale:.1f}× jump: expect invented texture; small text and faces may not match the source.")
    if ram_peak > mem["MemTotal"] - RAM_HEADROOM:
        blockers.append(f"Needs ≈{ram_peak / GB:.0f} GB RAM even with the smallest chunk – more than this machine can "
                        "spare. Use a lower resolution or the 3B model.")
    elif is_video and chunk <= b and frames > b:
        limits.append(f"Free RAM only fits {chunk} frames per chunk – close other apps (browser tabs) for faster runs.")
    if vram > vram_total - 0.5:
        (blockers if swap < 36 else limits).append(
            f"Estimated ≈{vram:.1f} GB VRAM of {vram_total:.0f} GB – risk of GPU out-of-memory. "
            + ("Raise blocks to swap to 36." if swap < 36 else "Use a smaller batch size or lower resolution."))
    if out_bytes + tmp_bytes > disk_free - 2 * GB:
        blockers.append(f"Needs ≈{(out_bytes + tmp_bytes) / GB:.1f} GB disk, only {disk_free / GB:.1f} GB free.")
    if is_video:
        if b == 1:
            limits.append("Batch size 1 processes frames independently – expect flicker. Use 5 or more.")
        if t > 3600:
            limits.append(f"Long run (≈{fmt_s(t)}). It's resumable, but try a frame cap of 33–65 first to check the look.")
        if mp > 8:
            limits.append("4K video is about 4× slower per frame than 1080p and output files get large.")
        if latent * mp > 12:
            limits.append("Large batch × resolution: each batch holds a lot of video in VRAM; lower batch if it fails.")

    return {
        "out_w": out_w, "out_h": out_h, "megapixels": round(mp, 2), "scale": round(scale, 2), "frames": frames,
        "time_s": round(t), "time": fmt_s(t), "sec_per_frame": round(t / frames, 1) if is_video else None,
        "model_load_s": round(load), "cold_load": cold,
        "chunk_size": chunk if is_video else None, "chunks": chunks if is_video else None,
        "batches": batches if is_video else None, "temporal_overlap": overlap if is_video else None,
        "ram_peak_gb": round(ram_peak / GB, 1), "ram_model_gb": round(model_ram / GB, 1),
        "ram_free_gb": round(usable_ram() / GB, 1), "ram_total_gb": round(mem["MemTotal"] / GB, 1),
        "vram_gb": round(vram, 1), "vram_total_gb": round(vram_total, 1),
        "output_gb": round(out_bytes / GB, 2), "temp_disk_gb": round(tmp_bytes / GB, 2),
        "disk_free_gb": round(disk_free / GB, 1),
        "limits": limits, "blockers": blockers, "ok": not blockers,
    }


# ---------- metadata ----------

def probe(path: Path) -> dict:
    """File metadata via ffprobe (works for both images and videos)."""
    meta = {"file": path.name, "size_mb": round(path.stat().st_size / 1e6, 3)}
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, timeout=30,
        ).stdout
        info = json.loads(out)
    except Exception as e:  # noqa: BLE001
        meta["probe_error"] = str(e)
        return meta
    v = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
    fmt = info.get("format", {})
    meta.update({
        "width": v.get("width"),
        "height": v.get("height"),
        "megapixels": round(v["width"] * v["height"] / 1e6, 3) if v.get("width") else None,
        "codec": v.get("codec_name"),
        "pix_fmt": v.get("pix_fmt"),
    })
    if path.suffix.lower() in VIDEO_EXT:
        num, _, den = (v.get("avg_frame_rate") or "0/1").partition("/")
        fps = float(num) / float(den or 1) if float(den or 1) else 0
        meta.update({
            "fps": round(fps, 3),
            "frames": int(v["nb_frames"]) if v.get("nb_frames", "").isdigit() else None,
            "duration_s": round(float(fmt.get("duration", 0)), 3),
            "bitrate_kbps": round(int(fmt.get("bit_rate", 0)) / 1000) if fmt.get("bit_rate") else None,
        })
    return meta


# ---------- quality analysis ----------

def tier_name(short: int) -> str:
    name = "below 240p"
    for px, label in STANDARD_TIERS:
        if short >= px * 0.95:
            name = label
    return name


def sample_frames(path: Path, is_video: bool, n: int = 8) -> list:
    import cv2
    import numpy as np
    if not is_video:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return [img] if img is not None else []
    cap = cv2.VideoCapture(str(path))
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    frames = []
    for i in np.linspace(0, total - 1, min(n, total)).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    return frames


def frame_stats(frame) -> tuple[float, float, float]:
    """(sharpness, noise sigma, blockiness) for one BGR frame. Heuristics, not ground truth."""
    import cv2
    import numpy as np
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
    sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
    # Immerkaer noise estimate, measured only on the flattest half of the frame so texture isn't counted as noise
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    resp = np.abs(cv2.filter2D(g, -1, k))[1:-1, 1:-1]
    grad = np.hypot(cv2.Sobel(g, cv2.CV_64F, 1, 0), cv2.Sobel(g, cv2.CV_64F, 0, 1))[1:-1, 1:-1]
    flat = grad <= np.percentile(grad, 50)
    noise = float(resp[flat].mean() * math.sqrt(math.pi / 2) / 6) if flat.any() else 0.0

    # Blockiness: gradient strength on the 8px codec grid vs everywhere else (1.0 = no blocking)
    def grid_ratio(d):
        if d.shape[1] < 16:
            return 1.0
        other = np.delete(d, np.s_[7::8], axis=1).mean()
        return float(d[:, 7::8].mean() / other) if other > 0 else 1.0
    block = (grid_ratio(np.abs(np.diff(g, axis=1))) + grid_ratio(np.abs(np.diff(g, axis=0)).T)) / 2
    return sharp, noise, block


def even(x: float) -> int:
    return max(2, int(round(x / 2)) * 2)


def upscale_targets(w: int, h: int, is_video: bool) -> list[dict]:
    short = min(w, h)
    opts = [dict(res=max(16, int(round(short / 8)) * 8), label="Same size (restore only)")]
    opts += [dict(res=px, label=label) for px, label in STANDARD_TIERS if px > short * 1.05]
    for t in opts:
        s = t["res"] / short
        t["scale"] = round(s, 2)
        t["out_w"], t["out_h"] = even(w * s), even(h * s)
        mp = t["out_w"] * t["out_h"] / 1e6
        t["megapixels"] = round(mp, 2)
        if s <= 1.05:
            t["tag"], t["note"] = "fix", "Cleans blur / noise / compression without enlarging"
        elif s <= 2.5:
            t["tag"], t["note"] = "good", "Safe jump – detail stays believable"
        elif s <= 4:
            t["tag"], t["note"] = "ok", "Big jump – model invents more detail"
        else:
            t["tag"], t["note"] = "risky", "Very large jump – can look artificial; better as two passes"
        if mp > 8.5:
            t["warn"] = "Very heavy on 16 GB VRAM (keep VAE tiling on)"
        elif is_video and mp > 2.2:
            t["warn"] = "Slow for video with 7B – test with a frame cap first"
    # Tiny sources tolerate a bigger first jump; otherwise stay within ~2.5x
    limit = 4 if short < 360 else 2.5
    ups = [t for t in opts if 1.05 < t["scale"] <= limit]
    rec = ups[-1] if ups else (opts[1] if len(opts) > 1 else opts[0])
    rec["recommended"] = True
    return opts


DEFAULT_PARAMS = dict(dit_model="seedvr2_ema_7b_fp16.safetensors", max_resolution=0, batch_size=5,
                      blocks_to_swap=36, vae_tiled=True, load_cap=0, chunk_size=0)


def capabilities(meta: dict, is_video: bool, short: int) -> list[str]:
    """What this input can be turned into, in plain words."""
    local = {p.name for p in (MODEL_DIR / "SEEDVR2").glob("*")}
    caps = ["Restore at the same size: removes blur, noise and compression blocks without enlarging.",
            "Upscale to any standard tier above the source (options below, with time and resource cost).",
            "Long-edge cap keeps portrait / ultra-wide sources from becoming huge."]
    if is_video:
        caps += [f"Any length: processed in RAM-sized chunks, saved as it goes, resumable after a stop or crash "
                 f"({meta.get('frames') or '?'} frames here).",
                 "Quick look first: set 'Video frames' to 17–65 to judge quality in a minute or two.",
                 "Original audio and frame rate are kept in the output.",
                 "Batch size trades speed for temporal smoothness: 5 is a good default; 13–21 is smoother but needs more VRAM."]
    caps.append("Heavy compression: add extra args '--input_noise_scale 0.1' (or up to 0.3) to let the model clean harder.")
    if "seedvr2_ema_7b_sharp_fp16.safetensors" not in local:
        caps.append("Sharper look: the 7B 'sharp' variant (≈16.5 GB download) gives crisper detail.")
    if not any("3b" in n for n in local):
        caps.append("Faster / lighter: the 3B model (≈7 GB download) runs ~2× faster and needs ~10 GB less RAM, "
                    "at somewhat lower quality.")
    return caps


def analyze(path: Path, with_targets: bool = True) -> dict:
    """Describe input quality and suggest standard upscale targets."""
    try:
        import numpy as np
    except ImportError:
        return {}
    is_video = path.suffix.lower() in VIDEO_EXT
    meta = probe(path)
    w, h = meta.get("width"), meta.get("height")
    if not w or not h:
        return {"meta": meta}
    frames = sample_frames(path, is_video)
    stats = np.array([frame_stats(f) for f in frames]) if frames else np.zeros((1, 3))
    sharp, noise, block = (float(x) for x in stats.mean(axis=0))
    short = min(w, h)

    if is_video and meta.get("bitrate_kbps") and meta.get("fps"):
        bpp = meta["bitrate_kbps"] * 1000 / (w * h * meta["fps"])
        comp = "heavy" if bpp < 0.05 else "moderate" if bpp < 0.1 else "light"
    else:
        bpp = path.stat().st_size * 8 / (w * h)
        comp = "none (lossless)" if path.suffix.lower() in {".png", ".bmp", ".tif", ".tiff"} else \
            "heavy" if bpp < 1 else "moderate" if bpp < 2.5 else "light"

    issues = []
    if short < 720:
        issues.append(f"Low resolution ({tier_name(short)})")
    if comp == "heavy":
        issues.append("Heavily compressed (low bitrate)")
    if block > 1.25:
        issues.append("Visible compression blocks")
    elif block > 1.1:
        issues.append("Mild compression blocks")
    if noise > 5:
        issues.append("Noisy / grainy")
    if sharp < 50:
        issues.append("Blurry / soft")
    overall = "Good" if not issues else "Fair" if len(issues) <= 2 else "Poor"

    out = {
        "tier": tier_name(short),
        "overall": overall,
        "issues": issues,
        "short_side": short,
        "orientation": "portrait" if h > w else "landscape" if w > h else "square",
        "compression": comp,
        "bits_per_pixel": round(bpp, 3),
        "sharpness": round(sharp, 1),
        "sharpness_label": "blurry" if sharp < 50 else "soft" if sharp < 200 else "sharp",
        "noise_sigma": round(noise, 2),
        "noise_label": "clean" if noise < 2 else "slight" if noise < 5 else "noisy",
        "blockiness": round(block, 3),
        "frames_sampled": len(frames),
        "meta": meta,
    }
    if with_targets:
        out["targets"] = upscale_targets(w, h, is_video)
        for t in out["targets"]:
            t["estimate"] = estimate(meta, is_video, dict(DEFAULT_PARAMS, resolution=t["res"]))
        out["capabilities"] = capabilities(meta, is_video, short)
        rec = next(t for t in out["targets"] if t.get("recommended"))
        tips = [f"Upscale to {rec['label']} ({rec['out_w']}×{rec['out_h']}, {rec['scale']}×)."
                if rec["scale"] > 1.05 else "Already high resolution – use 'Same size' to clean it up."]
        if comp == "heavy" or block > 1.1:
            tips.append("Compression artifacts present: SeedVR2 will clean them; if blocks remain, "
                        "try extra args '--input_noise_scale 0.1'.")
        if noise > 5:
            tips.append("Noisy source: keep color correction 'lab' or try 'wavelet' to avoid colour shifts.")
        if is_video:
            tips.append("Video: keep batch size ≥ 5 to avoid flicker, and test with a frame cap of 9–17 first.")
        out["advice"] = tips
    return out


# ---------- job store (survives crashes / restarts) ----------
# Each job lives in RUNS_DIR/<id>/: job.json (state, rewritten atomically), log.txt, input.*, outputs.
# Video jobs make the CLI write PNG frames chunk by chunk; a packer thread turns finished frames into
# small mp4 segments and deletes the PNGs. Segments on disk are the source of truth for progress, so a
# crashed or interrupted job resumes from the last packed frame instead of starting over.

ACTIVE = {"queued", "running", "finalizing"}
RESUMABLE = {"error", "interrupted", "stopped"}
MAX_AUTO_RESUME = 2       # automatic resumes after a server restart before waiting for the user
SEGMENT_FRAMES = 48       # PNG frames per mp4 segment; bounds the disk used by raw frames
SEG_RE = re.compile(r"seg_(\d{6})_(\d{4})\.mp4$")
CHUNK_RE = re.compile(r"Chunk (\d+)/(\d+)")
PHASE_RE = re.compile(r"Phase (\d): ([A-Za-z ]+?)\s*━")
STEP_RE = re.compile(r"(Encoding|Upscaling|Decoding|Post-processing) batch (\d+)/(\d+)")

jobs: dict[str, dict] = {}
jobs_lock = threading.RLock()
work_q: "queue.Queue[str]" = queue.Queue()
procs: dict[str, subprocess.Popen] = {}
worker_idle = threading.Event()
worker_idle.set()
shutting_down = threading.Event()


def job_dir(job_id: str) -> Path:
    return RUNS_DIR / job_id


def save_job(job: dict):
    with jobs_lock:
        data = json.dumps(job, indent=2, default=str)
    tmp = job_dir(job["id"]) / "job.json.tmp"
    tmp.write_text(data)
    tmp.replace(tmp.with_name("job.json"))


def update(job: dict, **kw):
    with jobs_lock:
        job.update(kw)
    save_job(job)


def set_progress(job: dict, **kw):
    with jobs_lock:
        job.setdefault("progress", {}).update(kw)


def log(job: dict, line: str):
    with (job_dir(job["id"]) / "log.txt").open("a") as f:
        f.write(f"[ui {time.strftime('%H:%M:%S')}] {line}\n")


def tail_log(job_id: str, n: int = 200) -> str:
    p = job_dir(job_id) / "log.txt"
    if not p.exists():
        return ""
    with p.open("rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 200_000))
        return "\n".join(f.read().decode(errors="replace").splitlines()[-n:])


def count_frames(path: Path) -> int:
    meta = probe(path)
    if meta.get("frames"):
        return meta["frames"]
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                          "stream=nb_read_packets", "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout
    return int(out.strip() or 0) if out.strip().isdigit() else int((meta.get("duration_s") or 0) * (meta.get("fps") or 0))


def frame_rate(path: Path) -> str:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.strip()
    return out if re.fullmatch(r"\d+/\d+", out) and not out.endswith("/0") else "25"


# ---------- video segments ----------

def segments(job: dict) -> list[tuple[int, int, Path]]:
    """Contiguous (start, count, path) segments from frame 0; stray or overlapping files are ignored."""
    segs = sorted((int(m[1]), int(m[2]), p) for p in (job_dir(job["id"]) / "segments").glob("seg_*.mp4")
                  if (m := SEG_RE.search(p.name)))
    out, nxt = [], 0
    for start, n, p in segs:
        if start == nxt:
            out.append((start, n, p))
            nxt += n
    return out


def frames_packed(job: dict) -> int:
    return sum(n for _, n, _ in segments(job))


def png_index(p: Path) -> int:
    return int(p.stem.rsplit("_", 1)[1])


def pack_frames(job: dict, att: Path, start: int, final: bool) -> int:
    """Encode finished PNGs of one CLI attempt into mp4 segments. Returns frames finished (packed + waiting)."""
    import cv2
    files = sorted(att.glob("*.png"), key=png_index) if att.exists() else []
    if not final:
        files = files[:-1]  # newest file may still be being written
    elif files and cv2.imread(str(files[-1])) is None:
        files[-1].unlink()  # truncated by a crash
        files = files[:-1]
    done = frames_packed(job)
    seg_dir = job_dir(job["id"]) / "segments"
    fps = job["params"]["fps_str"]
    while files and (len(files) >= SEGMENT_FRAMES or final):
        batch = files[:SEGMENT_FRAMES]
        first = png_index(batch[0])
        # keep only a gap-free run that continues exactly where the packed segments end
        run = [p for i, p in enumerate(batch) if png_index(p) == first + i]
        if start + first != done or not run:
            log(job, f"frame gap at {start + first} (expected {done}); stopping pack")
            break
        tmp = seg_dir / "seg_tmp.mp4"
        r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", fps, "-start_number", str(first),
                            "-i", str(att / f"{batch[0].stem.rsplit('_', 1)[0]}_%06d.png"), "-frames:v", str(len(run)),
                            "-c:v", "libx264", "-preset", "medium", "-crf", "14", "-pix_fmt", "yuv420p", str(tmp)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            log(job, f"segment encode failed: {r.stderr.strip()[-300:]}")
            break
        tmp.replace(seg_dir / f"seg_{done:06d}_{len(run):04d}.mp4")
        for p in run:
            p.unlink()
        done += len(run)
        files = files[len(run):]
    return done + len(files)


class Packer(threading.Thread):
    """Packs PNGs into segments while the CLI is running and keeps progress / job.json fresh."""

    def __init__(self, job: dict, att: Path, start: int):
        super().__init__(daemon=True)
        self.job, self.att, self.start_frame, self.stop = job, att, start, threading.Event()

    def run(self):
        while not self.stop.wait(3):
            try:
                set_progress(self.job, frames_done=pack_frames(self.job, self.att, self.start_frame, final=False))
                save_job(self.job)
            except Exception as e:  # noqa: BLE001
                log(self.job, f"packer error: {e!r}")


# ---------- running the CLI ----------

class VramPoller(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak_mb, self.stop = 0, threading.Event()

    def run(self):
        while not self.stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout.split()
                self.peak_mb = max(self.peak_mb, int(out[0]))
            except Exception:  # noqa: BLE001
                pass
            self.stop.wait(0.5)


def build_cmd(job: dict, inp: Path, out: Path, extra: list[str]) -> list[str]:
    p, is_video = job["params"], job["type"] == "video"
    cmd = [sys.executable, str(CLI), str(inp), "--output", str(out), "--output_format", "png",
           "--model_dir", str(MODEL_DIR / "SEEDVR2"), "--dit_model", p["dit_model"],
           "--resolution", str(p["resolution"]), "--max_resolution", str(p["max_resolution"]),
           "--batch_size", str(p["batch_size"] if is_video else 1), "--seed", str(p["seed"]),
           "--color_correction", p["color_correction"]]
    if p["blocks_to_swap"] > 0:
        cmd += ["--blocks_to_swap", str(p["blocks_to_swap"]), "--dit_offload_device", "cpu",
                "--vae_offload_device", "cpu"]
    if p["vae_tiled"]:
        cmd += ["--vae_encode_tiled", "--vae_decode_tiled"]
    return cmd + extra + (p["extra_args"].split() if p["extra_args"].strip() else [])


def parse_line(job: dict, line: str):
    if m := CHUNK_RE.search(line):
        set_progress(job, chunk=int(m[1]), chunks=int(m[2]))
    elif m := PHASE_RE.search(line):
        set_progress(job, phase=f"{m[1]}/4 {m[2]}", step=None)
    elif m := STEP_RE.search(line):
        set_progress(job, step=f"{m[1]} batch {m[2]}/{m[3]}")
    elif "Materializing DiT" in line or "Materializing VAE" in line:
        set_progress(job, step="loading model into RAM")


def run_cli(job: dict, cmd: list[str]) -> int:
    cmd, limit = memory_capped(cmd)
    update(job, cmd=" ".join(cmd), ram_limit_gb=round(limit / GB, 1) if limit else None)
    log(job, "$ " + " ".join(cmd))
    t0 = time.time()
    poller = VramPoller()
    poller.start()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=CLI.parent,
                            env=os.environ.copy(), bufsize=1, start_new_session=True)
    procs[job["id"]] = proc
    update(job, pid=proc.pid)
    last_save = 0.0
    with (job_dir(job["id"]) / "log.txt").open("a") as lf:
        for line in proc.stdout:  # universal newlines also splits tqdm's \r updates
            lf.write(line)
            lf.flush()
            parse_line(job, line)
            if time.time() - last_save > 2:
                save_job(job)
                last_save = time.time()
    proc.wait()
    poller.stop.set()
    procs.pop(job["id"], None)
    st = job.setdefault("stats", {})
    st["elapsed_s"] = round(st.get("elapsed_s", 0) + time.time() - t0, 2)
    st["peak_gpu_vram_mb"] = max(st.get("peak_gpu_vram_mb", 0), poller.peak_mb)
    st["exit_code"] = proc.returncode
    update(job, pid=None)
    return proc.returncode


def fail(job: dict, rc: int, extra: str = ""):
    if job.get("clean_shutdown"):
        save_job(job)
        return
    if job.get("stop"):
        update(job, status="stopped", error=None, stop=False)
        log(job, "stopped" + extra)
        return
    if rc in (-9, 137):
        msg = (f"Killed for using too much RAM (limit {job.get('ram_limit_gb')} GB). "
               "Chunk size was halved for the next resume.")
    else:
        msg = f"CLI exited with code {rc} – see log."
    update(job, status="error", error=msg + extra)


def image_metrics(inp: Path, out: Path) -> dict:
    """PSNR/SSIM of output vs bicubic-upscaled input + a diff heatmap. Rough sanity numbers, not a quality score."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return {}
    a = cv2.imread(str(inp), cv2.IMREAD_COLOR)
    b = cv2.imread(str(out), cv2.IMREAD_COLOR)
    if a is None or b is None:
        return {}
    a_up = cv2.resize(a, (b.shape[1], b.shape[0]), interpolation=cv2.INTER_CUBIC)
    ga, gb = cv2.cvtColor(a_up, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(ga, gb)
    heat = cv2.applyColorMap(cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX), cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(out.parent / "diff.png"), heat)
    cv2.imwrite(str(out.parent / "bicubic.png"), a_up)

    def sharpness(g):  # variance of Laplacian
        return round(float(cv2.Laplacian(g, cv2.CV_64F).var()), 1)

    # SSIM (single-scale, gaussian window)
    x, y = ga.astype(np.float64), gb.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda m: cv2.GaussianBlur(m, (11, 11), 1.5)  # noqa: E731
    mx, my = blur(x), blur(y)
    sx, sy, sxy = blur(x * x) - mx ** 2, blur(y * y) - my ** 2, blur(x * y) - mx * my
    ssim = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sx + sy + c2))
    return {
        "psnr_vs_bicubic_db": round(float(cv2.PSNR(a_up, b)), 2),
        "ssim_vs_bicubic": round(float(ssim.mean()), 4),
        "sharpness_bicubic": sharpness(ga),
        "sharpness_output": sharpness(gb),
        "mean_abs_diff": round(float(diff.mean()), 2),
    }


def add_audio(inp: Path, out_path: Path):
    """The CLI writes video only; copy the source audio track back in (trimmed to the processed range)."""
    has_audio = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                                "stream=index", "-of", "csv=p=0", str(inp)],
                               capture_output=True, text=True).stdout.strip()
    if not has_audio:
        return
    tmp = out_path.with_name("output_audio.mp4")
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(out_path), "-i", str(inp),
                        "-map", "0:v", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                        "-shortest", str(tmp)])
    if r.returncode == 0 and tmp.exists():
        tmp.replace(out_path)


def finish(job: dict, inp: Path, out_path: Path):
    in_meta, out_meta = probe(inp), probe(out_path)
    st = job["stats"]
    if in_meta.get("width") and out_meta.get("width"):
        st["scale_x"] = round(out_meta["width"] / in_meta["width"], 3)
    update(job, status="finalizing", input_meta=in_meta, output_meta=out_meta,
           output_url=f"/runs/{job['id']}/{out_path.name}")
    set_progress(job, phase="Measuring quality", step=None)
    job.update(input_quality=analyze(inp, with_targets=False), output_quality=analyze(out_path, with_targets=False))
    update(job, status="done", finished=time.time(), error=None)
    (out_path.parent / "result.json").write_text(json.dumps(
        {k: job.get(k) for k in ("params", "cmd", "stats", "input_meta", "output_meta", "input_quality", "output_quality")},
        indent=2))


def run_image(job: dict):
    d = job_dir(job["id"])
    inp, out = d / job["input_name"], d / "output.png"
    update(job, status="running", started=job.get("started") or time.time())
    rc = run_cli(job, build_cmd(job, inp, out, []))
    if rc != 0 or not out.exists():
        return fail(job, rc)
    job["stats"].update(image_metrics(inp, out))
    if (d / "diff.png").exists():
        job.update(diff_url=f"/runs/{job['id']}/diff.png", bicubic_url=f"/runs/{job['id']}/bicubic.png")
    finish(job, inp, out)


def run_video(job: dict):
    d, p = job_dir(job["id"]), job["params"]
    inp, out = d / job["input_name"], d / "output.mp4"
    frames_root = d / "frames"
    (d / "segments").mkdir(exist_ok=True)
    frames_root.mkdir(exist_ok=True)
    update(job, status="running", started=job.get("started") or time.time(), error=None)

    # Pack whatever a previous (crashed) attempt left behind, then continue after it
    for att in sorted(frames_root.glob("a*")):
        pack_frames(job, att / inp.stem, int(att.name[1:]), final=True)
        shutil.rmtree(att, ignore_errors=True)
    total, done = p["frames_total"], frames_packed(job)
    set_progress(job, frames_done=done, frames_total=total, chunk=None, chunks=None, phase=None, step=None,
                 attempt_start_frame=done, attempt_started=time.time())
    if done:
        log(job, f"resuming at frame {done}/{total}")

    if p.get("chunk_auto") and not p.get("chunk_shrunk"):
        p["chunk_size"] = auto_chunk(p["out_w"], p["out_h"], p["batch_size"], p["dit_model"])
        p["chunks"] = math.ceil(max(1, total - done) / p["chunk_size"])
        log(job, f"auto chunk size: {p['chunk_size']} frames")
    if done < total:
        att = frames_root / f"a{done:06d}"
        extra = ["--skip_first_frames", str(done), "--load_cap", str(total - done),
                 "--chunk_size", str(p["chunk_size"]), "--temporal_overlap", str(p["temporal_overlap"])]
        if p["blocks_to_swap"] > 0:  # keep models in RAM between chunks instead of reloading 16 GB each time
            extra += ["--cache_dit", "--cache_vae"]
        png_dir = att / inp.stem  # the CLI writes video PNGs to <--output>/<input stem>/
        packer = Packer(job, png_dir, done)
        packer.start()
        rc = run_cli(job, build_cmd(job, inp, att, extra))
        packer.stop.set()
        packer.join()
        done = pack_frames(job, png_dir, done, final=True)
        shutil.rmtree(att, ignore_errors=True)
        set_progress(job, frames_done=done)
        if rc != 0:
            if rc in (-9, 137):
                b = p["batch_size"]
                p["chunk_size"] = max(b, p["chunk_size"] // 2 // b * b)
                p["chunk_shrunk"] = True
            return fail(job, rc, f" {done}/{total} frames are saved; Resume continues from there.")
        if done < total:  # container frame count was an over-estimate
            log(job, f"source ended at {done} frames (expected {total})")
            p["frames_total"] = done

    if not segments(job):
        return update(job, status="error", error="No frames were produced – see log.")
    # Join segments, put the audio back, clean up intermediate files
    update(job, status="finalizing")
    set_progress(job, phase="Joining segments + audio", step=None, chunk=None)
    lst = d / "segments.txt"
    lst.write_text("".join(f"file '{s.relative_to(d)}'\n" for _, _, s in segments(job)))
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-c", "copy", "-movflags", "+faststart", str(out)], cwd=d, capture_output=True, text=True)
    if r.returncode != 0:
        return update(job, status="error", error=f"Joining segments failed: {r.stderr.strip()[-300:]}")
    add_audio(inp, out)
    shutil.rmtree(d / "segments", ignore_errors=True)
    shutil.rmtree(frames_root, ignore_errors=True)
    lst.unlink(missing_ok=True)

    in_meta = probe(inp)
    job["input_url"] = f"/runs/{job['id']}/{inp.name}"
    if not (in_meta.get("codec") == "h264" and in_meta.get("pix_fmt") == "yuv420p"):
        iprev = d / "input_preview.mp4"  # browsers can't play every source codec
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(inp), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "16", "-an", str(iprev)])
        job["input_url"] = f"/runs/{job['id']}/{iprev.name}"
    out_frames = probe(out).get("frames")
    if out_frames:
        job["stats"]["sec_per_frame"] = round(job["stats"]["elapsed_s"] / out_frames, 3)
    finish(job, inp, out)


def worker():
    """Single GPU worker: jobs run one at a time in submission order."""
    while True:
        job = jobs.get(work_q.get())
        if not job or job["status"] != "queued" or shutting_down.is_set():
            continue  # queued jobs stay queued on disk and are picked up again on next start
        worker_idle.clear()
        try:
            (run_video if job["type"] == "video" else run_image)(job)
        except Exception as e:  # noqa: BLE001
            log(job, f"UI error: {e!r}")
            update(job, status="error", error=f"UI error: {e}")
        finally:
            worker_idle.set()


def enqueue(job: dict):
    update(job, status="queued", queued_at=time.time())
    work_q.put(job["id"])


# ---------- startup recovery ----------

def kill_orphan(job: dict):
    """A CLI process left over from a killed server would fight the resumed job for the GPU."""
    pid = job.get("pid")
    try:
        if pid and job["id"] in Path(f"/proc/{pid}/cmdline").read_text():
            os.killpg(pid, 15)
            for _ in range(20):
                time.sleep(0.5)
                os.kill(pid, 0)
            os.killpg(pid, 9)
    except (OSError, ProcessLookupError):
        pass


def legacy_job(d: Path) -> dict | None:
    """Runs made before job.json existed: show them in history from result.json / leftover files."""
    inp = next(d.glob("input.*"), None)
    if inp is None:
        return None
    is_video = inp.suffix.lower() in VIDEO_EXT
    job = {"id": d.name, "type": "video" if is_video else "image", "input_name": inp.name, "legacy": True,
           "created": d.stat().st_mtime, "input_url": f"/runs/{d.name}/{inp.name}", "stats": {}}
    if (d / "result.json").exists():
        r = json.loads((d / "result.json").read_text())
        out = next((x for x in ("output_preview.mp4", "output.mp4", "output.png") if (d / x).exists()), None)
        job.update({k: r.get(k) for k in ("params", "cmd", "stats", "input_meta", "output_meta",
                                          "input_quality", "output_quality")}, status="done", output_url=f"/runs/{d.name}/{out}")
        if (d / "input_preview.mp4").exists():
            job["input_url"] = f"/runs/{d.name}/input_preview.mp4"
        if (d / "diff.png").exists():
            job.update(diff_url=f"/runs/{d.name}/diff.png", bicubic_url=f"/runs/{d.name}/bicubic.png")
    else:
        job.update(status="interrupted", params={"file": inp.name},
                   error="Crashed before job tracking existed – settings unknown. Use 'Load input' to run it again.")
    return job


def load_jobs():
    for d in sorted(RUNS_DIR.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        try:
            job = json.loads((d / "job.json").read_text()) if (d / "job.json").exists() else legacy_job(d)
        except Exception:  # noqa: BLE001
            continue
        if job:
            jobs[job["id"]] = job
    for job in sorted(jobs.values(), key=lambda j: j.get("queued_at") or j.get("created") or 0):
        if job["status"] not in ACTIVE or job.get("legacy"):
            continue
        kill_orphan(job)
        job["pid"] = None
        clean = job.pop("clean_shutdown", False)
        if clean or job.get("auto_resumes", 0) < MAX_AUTO_RESUME:
            job["auto_resumes"] = job.get("auto_resumes", 0) + (0 if clean else 1)
            job["resumed"] = True
            log(job, "server restarted – resuming automatically")
            enqueue(job)
        else:
            update(job, status="interrupted",
                   error="Interrupted again after automatic resumes – press Resume to try once more.")


# ---------- API ----------

def system_stats() -> dict:
    m = mem_info()
    s = {"ram_total_gb": round(m["MemTotal"] / GB, 1), "ram_used_gb": round((m["MemTotal"] - m["MemAvailable"]) / GB, 1),
         "swap_used_gb": round((m["SwapTotal"] - m["SwapFree"]) / GB, 1)}
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        used, total, util, temp = (int(x) for x in out.split(","))
        s.update(vram_used_gb=round(used / 1024, 1), vram_total_gb=round(total / 1024, 1), gpu_util=util, gpu_temp=temp)
    except Exception:  # noqa: BLE001
        pass
    running = next((j for j in jobs.values() if j["status"] in ("running", "finalizing") and j.get("pid")), None)
    if running:
        try:
            rss = next(l for l in Path(f"/proc/{running['pid']}/status").read_text().splitlines() if l.startswith("VmRSS"))
            s["job_ram_gb"] = round(int(rss.split()[1]) * 1024 / GB, 1)
        except (OSError, StopIteration):
            pass
    return s


def job_view(job: dict) -> dict:
    j = json.loads(json.dumps(job, default=str))
    pr = j.get("progress") or {}
    if j["status"] in ("running", "finalizing") and pr.get("frames_total"):
        total, done = pr["frames_total"], pr.get("frames_done") or 0
        made = done - (pr.get("attempt_start_frame") or 0)
        spent = time.time() - (pr.get("attempt_started") or time.time())
        # finer progress inside the current chunk from phase + batch counters
        chunk_frames = min(j["params"].get("chunk_size") or total, total - done)
        frac = 0.0
        if (m := re.match(r"(\d)/4", pr.get("phase") or "")) and (s := re.search(r"(\d+)/(\d+)", pr.get("step") or "")):
            starts, weights = {1: 0, 2: .15, 3: .65, 4: .95}, {1: .15, 2: .5, 3: .3, 4: .05}
            ph = int(m[1])
            frac = starts[ph] + weights[ph] * int(s[1]) / max(1, int(s[2]))
        pr["percent"] = round(min(99.9, 100 * (done + frac * chunk_frames) / total), 1)
        if made > 0 and spent > 0:
            pr["sec_per_frame"] = round(spent / made, 2)
            pr["eta_s"] = round((total - done) * spent / made)
        elif (j.get("estimate") or {}).get("time_s"):
            est = j["estimate"]["time_s"] * (total - (pr.get("attempt_start_frame") or 0)) / total
            pr["eta_s"] = max(0, round(est - spent))
            pr["eta_is_estimate"] = True
    if j.get("queued_at") and j["status"] == "queued":
        j["queue_position"] = 1 + sum(1 for o in jobs.values() if o["status"] == "queued"
                                      and (o.get("queued_at") or 0) < j["queued_at"])
    j["resumable"] = j["status"] in RESUMABLE and not j.get("legacy")
    return j


@app.get("/api/models")
def models():
    sys.path.insert(0, str(CLI.parent))
    try:
        from src.utils.model_registry import get_available_dit_models  # type: ignore
        names = get_available_dit_models()
    except Exception:  # noqa: BLE001
        names = ["seedvr2_ema_7b_fp16.safetensors", "seedvr2_ema_7b_sharp_fp16.safetensors",
                 "seedvr2_ema_3b_fp16.safetensors", "seedvr2_ema_3b_fp8_e4m3fn.safetensors"]
    local = {p.name for p in (MODEL_DIR / "SEEDVR2").glob("*")} if (MODEL_DIR / "SEEDVR2").exists() else set()
    return [{"name": n, "downloaded": n in local} for n in names]


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """Store the file once and analyse it; runs reference it by upload_id so settings can be re-tried."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in IMAGE_EXT | VIDEO_EXT:
        raise HTTPException(400, f"Unsupported file type: {ext}")
    upload_id = uuid.uuid4().hex[:10]
    dst = UPLOADS_DIR / f"{upload_id}{ext}"
    with dst.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"upload_id": upload_id, "filename": file.filename, "analysis": analyze(dst)}


@app.post("/api/run")
def run(
    upload_id: str = Form(...),
    filename: str = Form(""),
    dit_model: str = Form("seedvr2_ema_7b_fp16.safetensors"),
    resolution: int = Form(1080),
    max_resolution: int = Form(0),
    batch_size: int = Form(5),
    seed: int = Form(42),
    color_correction: str = Form("lab"),
    blocks_to_swap: int = Form(36),
    vae_tiled: bool = Form(True),
    load_cap: int = Form(0),
    chunk_size: int = Form(0),
    extra_args: str = Form(""),
):
    src = next((p for p in UPLOADS_DIR.glob(f"{upload_id}.*") if upload_id.isalnum()), None)
    if src is None:
        raise HTTPException(404, "Upload not found – choose the file again")
    ext = src.suffix.lower()
    is_video = ext in VIDEO_EXT
    job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    run_dir = job_dir(job_id)
    run_dir.mkdir(parents=True)
    inp = run_dir / f"input{ext}"
    shutil.copy2(src, inp)

    params = dict(file=filename or src.name, type="video" if is_video else "image", dit_model=dit_model,
                  resolution=resolution, max_resolution=max_resolution, batch_size=batch_size if is_video else 1,
                  seed=seed, color_correction=color_correction, blocks_to_swap=blocks_to_swap, vae_tiled=vae_tiled,
                  load_cap=load_cap, extra_args=extra_args)
    if is_video:
        # Stream the video in RAM-bounded chunks; frames go to disk as soon as each chunk is done.
        meta = probe(inp)
        out_w, out_h = output_size(meta.get("width") or 1920, meta.get("height") or 1080, resolution, max_resolution)
        chunk_auto = chunk_size <= 0
        if chunk_auto:
            chunk_size = auto_chunk(out_w, out_h, batch_size, dit_model)
        total = count_frames(inp)
        if load_cap > 0:
            total = min(total, load_cap) if total else load_cap
        params.update(chunk_size=chunk_size, chunk_auto=chunk_auto, temporal_overlap=0 if batch_size < 5 else max(1, batch_size // 4),
                      out_w=out_w, out_h=out_h, frames_total=total, fps_str=frame_rate(inp),
                      chunks=math.ceil(total / chunk_size) if total else None)
    est = estimate(probe(inp), is_video, params)
    if not est["ok"]:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise HTTPException(400, " ".join(est["blockers"]))
    job = {"id": job_id, "type": params["type"], "status": "queued", "params": params, "input_name": inp.name,
           "estimate": est, "estimate_s": est["time_s"],
           "created": time.time(), "input_url": f"/runs/{job_id}/{inp.name}", "stats": {},
           "progress": {"frames_total": params.get("frames_total")}}
    jobs[job_id] = job
    log(job, f"created: {json.dumps(params)}")
    enqueue(job)
    return {"id": job_id}


@app.post("/api/estimate")
def api_estimate(
    upload_id: str = Form(...),
    dit_model: str = Form("seedvr2_ema_7b_fp16.safetensors"),
    resolution: int = Form(1080),
    max_resolution: int = Form(0),
    batch_size: int = Form(5),
    blocks_to_swap: int = Form(36),
    vae_tiled: bool = Form(True),
    load_cap: int = Form(0),
    chunk_size: int = Form(0),
):
    """Live plan for the current settings (time, RAM, VRAM, disk, limitations)."""
    src = next((p for p in UPLOADS_DIR.glob(f"{upload_id}.*") if upload_id.isalnum()), None)
    if src is None:
        raise HTTPException(404, "Upload not found")
    return estimate(probe(src), src.suffix.lower() in VIDEO_EXT, dict(
        dit_model=dit_model, resolution=resolution, max_resolution=max_resolution, batch_size=batch_size,
        blocks_to_swap=blocks_to_swap, vae_tiled=vae_tiled, load_cap=load_cap, chunk_size=chunk_size))


@app.get("/api/jobs")
def list_jobs():
    keys = ("id", "type", "status", "created", "finished", "error", "progress", "output_url", "input_url", "legacy")
    out = []
    for j in sorted(jobs.values(), key=lambda j: j.get("created") or 0, reverse=True):
        p = j.get("params") or {}
        out.append({**{k: j.get(k) for k in keys}, "file": p.get("file"), "resolution": p.get("resolution"),
                    "elapsed_s": (j.get("stats") or {}).get("elapsed_s")})
    return out


@app.get("/api/job/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404)
    j = job_view(jobs[job_id])
    j["log"] = tail_log(job_id)
    return j


@app.get("/api/system")
def system():
    return system_stats()


def stop_proc(job_id: str):
    """SIGTERM the CLI's process group; finished frames are packed afterwards, so nothing done is lost."""
    proc = procs.get(job_id)
    if proc and proc.poll() is None:
        os.killpg(proc.pid, 15)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, 9)


@app.post("/api/job/{job_id}/stop")
def stop(job_id: str):
    job = jobs.get(job_id) or {}
    if job.get("status") == "queued":
        update(job, status="stopped")
    elif job.get("status") == "running":
        job["stop"] = True
        threading.Thread(target=stop_proc, args=(job_id,), daemon=True).start()
    else:
        raise HTTPException(400, "Job is not running")
    log(job, "stop requested")
    return {"ok": True}


@app.on_event("shutdown")
def on_shutdown():
    """Server stopping (Ctrl+C): end the CLI cleanly and leave the job marked for auto-resume on next start."""
    shutting_down.set()
    for job_id in list(procs):
        job = jobs.get(job_id)
        if job:
            job["clean_shutdown"] = True
            log(job, "server shutting down – job will resume on next start")
            save_job(job)
        stop_proc(job_id)
    worker_idle.wait(60)  # let the worker pack finished frames before the interpreter exits


@app.post("/api/job/{job_id}/resume")
def resume(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] not in RESUMABLE or job.get("legacy"):
        raise HTTPException(400, "Job can't be resumed")
    job["auto_resumes"] = 0
    job["resumed"] = True
    log(job, "resume requested")
    enqueue(job)
    return {"ok": True}


@app.post("/api/job/{job_id}/reuse")
def reuse(job_id: str):
    """Load a past job's input as a fresh upload so it can be re-run with new settings."""
    job = jobs.get(job_id)
    inp = job_dir(job_id) / job["input_name"] if job else None
    if not inp or not inp.exists():
        raise HTTPException(404, "Input file is gone")
    upload_id = uuid.uuid4().hex[:10]
    dst = UPLOADS_DIR / f"{upload_id}{inp.suffix.lower()}"
    shutil.copy2(inp, dst)
    return {"upload_id": upload_id, "filename": (job.get("params") or {}).get("file") or inp.name,
            "analysis": analyze(dst), "params": job.get("params")}


@app.delete("/api/job/{job_id}")
def delete(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404)
    if job["status"] in ACTIVE:
        raise HTTPException(400, "Cancel the job first")
    shutil.rmtree(job_dir(job_id), ignore_errors=True)
    jobs.pop(job_id, None)
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


app.mount("/runs", StaticFiles(directory=RUNS_DIR), name="runs")

if __name__ == "__main__":
    load_jobs()
    threading.Thread(target=worker, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 7860)))
