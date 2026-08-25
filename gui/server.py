#!/usr/bin/env python3
"""Local web GUI wrapping the h3 CLI binary.

Standard library only - no pip install needed. Runs h3 as a subprocess per
job, parses its uniform "\\r%-25s %4d/%-4d" progress lines (see cli_progress
in h3_cli.c) into JSON the browser polls, and serves the finished MP4 with
Range support so the <video> element can seek.

Usage: python3 gui/server.py [--port 8420]
"""
import argparse
import http.server
import json
import mimetypes
import os
import re
import socketserver
import subprocess
import threading
import time
import urllib.parse
import uuid
from pathlib import Path

GUI_DIR = Path(__file__).resolve().parent
REPO_DIR = GUI_DIR.parent
DEFAULT_MODEL_DIR = REPO_DIR.parent / "MiniMax-H3"
DEFAULT_ATTENTION_CACHE = REPO_DIR / "dit_int8_v2.cache"
H3_BINARY = REPO_DIR / "h3"
OUTPUT_DIR = GUI_DIR / "outputs"
STATIC_DIR = GUI_DIR / "static"

# H3_DIT_BLOCKS in h3_dit_schedule.h - --layers must fall in this range.
LAYERS_MIN, LAYERS_MAX = 35, 50

PROGRESS_RE = re.compile(r"(\S.*?)\s+(\d+)\s*/\s*(\d+)\s*$")

jobs = {}
jobs_lock = threading.Lock()


class Job:
    def __init__(self, job_id, argv, env, output_path):
        self.id = job_id
        self.argv = argv
        self.env = env
        self.output_path = output_path
        self.state = "queued"        # queued -> running -> done | error | cancelled
        self.phase = ""
        self.completed = 0
        self.total = 0
        self.log_tail = []
        self.error = None
        self.started_at = time.time()
        self.finished_at = None
        self.process = None

    def to_json(self):
        return {
            "id": self.id,
            "state": self.state,
            "phase": self.phase,
            "completed": self.completed,
            "total": self.total,
            "error": self.error,
            "elapsed": round((self.finished_at or time.time()) - self.started_at, 1),
            "log_tail": self.log_tail[-12:],
            "video_ready": self.state == "done",
        }


def run_job(job):
    job.state = "running"
    try:
        job.process = subprocess.Popen(
            job.argv, cwd=str(REPO_DIR), env=job.env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, universal_newlines=True,
        )
    except OSError as error:
        job.state = "error"
        job.error = f"cannot start h3: {error}"
        job.finished_at = time.time()
        return

    buffer = ""
    assert job.process.stdout is not None
    while True:
        chunk = job.process.stdout.read(1)
        if chunk == "":
            break
        if chunk in ("\r", "\n"):
            line = buffer.strip()
            buffer = ""
            if not line:
                continue
            match = PROGRESS_RE.match(line)
            if match:
                job.phase = match.group(1).strip()
                job.completed = int(match.group(2))
                job.total = int(match.group(3))
            else:
                job.log_tail.append(line)
                if line.startswith("h3: ") and "cannot" in line:
                    job.error = line
        else:
            buffer += chunk
    if buffer.strip():
        job.log_tail.append(buffer.strip())

    return_code = job.process.wait()
    job.finished_at = time.time()
    if return_code == 0 and job.output_path.exists():
        job.state = "done"
    else:
        job.state = "error"
        if not job.error:
            job.error = (job.log_tail[-1] if job.log_tail else
                         f"h3 exited with status {return_code}")


def align_frames(seconds):
    """Matches h3_align_frame_count: round up to 5 + 17*k."""
    requested = max(5, round(seconds * 24))
    remainder = (requested - 5) % 17
    if remainder:
        requested += 17 - remainder
    return requested


def build_job(params):
    prompt = (params.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("prompt is required")

    profile = params.get("profile", "square")
    if profile == "landscape_upscaled":
        width, height = 1344, 768
        render_width, render_height = 672, 384
    else:
        width, height = 512, 512
        render_width = render_height = None

    seconds = float(params.get("seconds", 5))
    frames = align_frames(seconds)

    layers = int(params.get("layers", 50))
    if not (LAYERS_MIN <= layers <= LAYERS_MAX):
        raise ValueError(f"layers must be between {LAYERS_MIN} and {LAYERS_MAX}")
    reuse = int(params.get("reuse", 2))
    if not (1 <= reuse <= 6):
        raise ValueError("reuse must be between 1 and 6")
    steps = int(params.get("steps", 20))
    if not (1 <= steps <= 100):
        raise ValueError("steps must be between 1 and 100")

    seed = params.get("seed")
    seed = int(seed) if seed not in (None, "") else None

    job_id = uuid.uuid4().hex[:12]
    output_path = OUTPUT_DIR / f"{job_id}.mp4"

    argv = [
        str(H3_BINARY), "-d", str(DEFAULT_MODEL_DIR), "-p", prompt,
        "--width", str(width), "--height", str(height),
        "--frames", str(frames), "--steps", str(steps),
        "--layers", str(layers), "--reuse", str(reuse),
        "-o", str(output_path),
    ]
    if render_width:
        argv += ["--render-width", str(render_width),
                 "--render-height", str(render_height)]
    if seed is not None:
        argv += ["--seed", str(seed)]

    env = dict(os.environ)
    env["H3_QWEN_PREFETCH_DEPTH"] = "1"
    if DEFAULT_ATTENTION_CACHE.exists():
        env["H3_ATTENTION_CACHE"] = str(DEFAULT_ATTENTION_CACHE)
        env["H3_INT8_STREAM_MLP"] = "1"

    return Job(job_id, argv, env, output_path)


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "H3GUI/1"
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        pass  # keep stdout clean; job state is visible via the API/UI

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, rel_path):
        path = (STATIC_DIR / rel_path).resolve()
        if STATIC_DIR not in path.parents and path != STATIC_DIR:
            self.send_error(403)
            return
        if not path.is_file():
            self.send_error(404)
            return
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_video(self, job):
        path = job.output_path
        if not path.is_file():
            self.send_error(404)
            return
        size = path.stat().st_size
        range_header = self.headers.get("Range")
        start, end = 0, size - 1
        status = 200
        if range_header and range_header.startswith("bytes="):
            status = 206
            spec = range_header.split("=", 1)[1].split("-")
            if spec[0]:
                start = int(spec[0])
            if len(spec) > 1 and spec[1]:
                end = int(spec[1])
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(path, "rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                block = handle.read(min(65536, remaining))
                if not block:
                    break
                self.wfile.write(block)
                remaining -= len(block)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        parts = [p for p in parsed.path.split("/") if p]

        if parsed.path == "/" or parsed.path == "":
            self._send_static("index.html")
        elif parts and parts[0] == "static":
            self._send_static("/".join(parts[1:]))
        elif parts[:2] == ["api", "jobs"] and len(parts) == 3:
            with jobs_lock:
                job = jobs.get(parts[2])
            if not job:
                self._send_json({"error": "unknown job"}, 404)
            else:
                self._send_json(job.to_json())
        elif parts[:2] == ["api", "jobs"] and len(parts) == 4 and parts[3] == "video":
            with jobs_lock:
                job = jobs.get(parts[2])
            if not job or job.state != "done":
                self.send_error(404)
            else:
                self._send_video(job)
        elif parsed.path == "/api/config":
            self._send_json({
                "model_dir": str(DEFAULT_MODEL_DIR),
                "model_dir_ok": DEFAULT_MODEL_DIR.exists(),
                "attention_cache_ok": DEFAULT_ATTENTION_CACHE.exists(),
                "layers_min": LAYERS_MIN,
                "layers_max": LAYERS_MAX,
            })
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/api/generate":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            params = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._send_json({"error": "invalid JSON body"}, 400)
            return
        try:
            job = build_job(params)
        except (ValueError, KeyError) as error:
            self._send_json({"error": str(error)}, 400)
            return
        with jobs_lock:
            jobs[job.id] = job
        threading.Thread(target=run_job, args=(job,), daemon=True).start()
        self._send_json({"id": job.id})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8420)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not H3_BINARY.is_file():
        raise SystemExit(f"h3 binary not found at {H3_BINARY} - run `make` first")
    if not DEFAULT_MODEL_DIR.is_dir():
        print(f"warning: model directory not found at {DEFAULT_MODEL_DIR}")
    if not DEFAULT_ATTENTION_CACHE.exists():
        print(f"warning: attention cache not found at {DEFAULT_ATTENTION_CACHE} "
              "- falling back to slower resident weight loading")

    class ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True

    server = ThreadingServer(("127.0.0.1", args.port), Handler)
    print(f"H3 GUI: http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
