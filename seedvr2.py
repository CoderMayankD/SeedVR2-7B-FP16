#!/usr/bin/env python3
"""Command-line / Python client for the SeedVR2 server (ui/app.py). Standard library only.

  python seedvr2.py photo.jpg                      # -> photo_4k.png next to it
  python seedvr2.py input/images -o output/4k      # every image in the folder
  python seedvr2.py clip.mp4 --target 1080p
  python seedvr2.py a.png --resolution 3000 --max-resolution 0

From Python:  from seedvr2 import upscale;  upscale("photo.jpg", "photo_4k.png", target="4k")
The server must be running (source ./env.sh && python ui/app.py); jobs run one at a time on the GPU.
"""
import argparse
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

SERVER = os.environ.get("SEEDVR2_URL", "http://127.0.0.1:7860")
MEDIA_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".mp4", ".mov", ".mkv", ".avi", ".webm"}


class SeedVR2Error(RuntimeError):
    pass


def _request(method: str, path: str, server: str, body: bytes | None = None, headers: dict | None = None):
    req = urllib.request.Request(server.rstrip("/") + path, data=body, method=method, headers=headers or {})
    try:
        return urllib.request.urlopen(req, timeout=600)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        try:
            detail = json.loads(detail).get("detail", detail)
        except (ValueError, AttributeError):
            pass
        raise SeedVR2Error(f"HTTP {e.code}: {detail}") from None
    except urllib.error.URLError as e:
        raise SeedVR2Error(f"Cannot reach SeedVR2 server at {server} ({e.reason}). "
                           "Start it with: source ./env.sh && python ui/app.py") from None


def submit(path: str | Path, target: str = "4k", resolution: int = 0, max_resolution: int = -1,
           server: str = SERVER, **options) -> dict:
    """Queue one file; returns the job (id, status, ...). options: dit_model, seed, color_correction, batch_size."""
    path = Path(path)
    fields = {"target": target, "resolution": resolution, "max_resolution": max_resolution, **options}
    boundary = uuid.uuid4().hex
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
             for k, v in fields.items()]
    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
                 f"Content-Type: {ctype}\r\n\r\n".encode() + path.read_bytes() + b"\r\n")
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()
    with _request("POST", "/api/v1/upscale", server, body,
                  {"Content-Type": f"multipart/form-data; boundary={boundary}"}) as r:
        return json.load(r)


def status(job_id: str, server: str = SERVER) -> dict:
    with _request("GET", f"/api/v1/jobs/{job_id}", server) as r:
        return json.load(r)


def download(job_id: str, out: str | Path, server: str = SERVER) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with _request("GET", f"/api/v1/jobs/{job_id}/result", server) as r, out.open("wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    return out


def wait(job_id: str, server: str = SERVER, poll: float = 2, on_progress=None) -> dict:
    """Block until the job is done; raises SeedVR2Error if it fails or is stopped."""
    while True:
        job = status(job_id, server)
        if on_progress:
            on_progress(job)
        if job["status"] == "done":
            return job
        if job["status"] not in ("queued", "running", "finalizing"):
            raise SeedVR2Error(f"job {job_id} {job['status']}: {job.get('error')}")
        time.sleep(poll)


def default_output(src: Path, out_dir: Path | None, label: str) -> Path:
    suffix = ".mp4" if src.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm"} else ".png"
    return (out_dir or src.parent) / f"{src.stem}_{label}{suffix}"


def upscale(path: str | Path, out: str | Path | None = None, target: str = "4k", server: str = SERVER,
            **kw) -> Path:
    """Submit, wait and download in one call. Returns the output path."""
    path = Path(path)
    job = submit(path, target, server=server, **kw)
    wait(job["id"], server)
    label = f"{kw['resolution']}p" if kw.get("resolution") else target.lower()
    return download(job["id"], out or default_output(path, None, label), server)


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
    ap = argparse.ArgumentParser(description="Upscale / restore images and videos with the local SeedVR2 server.")
    ap.add_argument("inputs", nargs="+", help="image/video files or folders")
    ap.add_argument("-o", "--out-dir", type=Path, help="output folder (default: next to each input)")
    ap.add_argument("-t", "--target", default="4k", help="720p, 1080p, 1440p/2k, 2160p/4k (default 4k)")
    ap.add_argument("--resolution", type=int, default=0, help="exact short side in px (overrides --target)")
    ap.add_argument("--max-resolution", type=int, default=-1, help="long-side cap in px, 0 = none")
    ap.add_argument("--model", dest="dit_model", help="DiT weights, e.g. seedvr2_ema_7b_sharp_fp16.safetensors")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--color-correction", dest="color_correction")
    ap.add_argument("--server", default=SERVER, help=f"server URL (default {SERVER}, env SEEDVR2_URL)")
    ap.add_argument("--skip-existing", action="store_true", help="skip inputs whose output file already exists")
    a = ap.parse_args()

    opts = {k: v for k, v in dict(dit_model=a.dit_model, seed=a.seed, color_correction=a.color_correction).items()
            if v is not None}
    label = f"{a.resolution}p" if a.resolution > 0 else a.target.lower()
    todo = []
    for src in collect(a.inputs):
        out = default_output(src, a.out_dir, label)
        if a.skip_existing and out.exists():
            print(f"skip  {src} (exists: {out})")
            continue
        todo.append((src, out))
    if not todo:
        return

    # Queue everything up front; the server runs them one by one on the GPU
    jobs = []
    try:
        for src, out in todo:
            job = submit(src, a.target, a.resolution, a.max_resolution, a.server, **opts)
            print(f"queued {src} -> job {job['id']}")
            jobs.append((src, out, job["id"]))
    except SeedVR2Error as e:
        sys.exit(f"error: {e}")

    failed = 0
    for src, out, job_id in jobs:
        last = [None]

        def show(j, src=src, last=last):
            msg = f"{j['status']}" + (f" {j['percent']}%" if j.get("percent") is not None else "") + \
                  (f" (queue #{j['queue_position']})" if j.get("queue_position") else "")
            if msg != last[0]:
                print(f"  {src.name}: {msg}", flush=True)
                last[0] = msg
        try:
            job = wait(job_id, a.server, on_progress=show)
            download(job_id, out, a.server)
            print(f"done  {src} -> {out} ({job['width']}x{job['height']}, {job['elapsed_s']} s)")
        except SeedVR2Error as e:
            failed += 1
            print(f"FAIL  {src}: {e}", file=sys.stderr)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
