import atexit
import hashlib
import json
import os
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import config

_JOBS_DIR = config.OUTPUT_DIR / "jobs"
_CACHE_DIR = config.OUTPUT_DIR / "cache"
_JOBS_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_lock = threading.Lock()
_running_jobs: dict[str, dict] = {}


def _sha256(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _now_iso() -> str:
    return datetime.now().isoformat()


# ─── Scene Cache ────────────────────────────────────────────────────────────


def cache_key(prompt: str, model: str, duration: int = 5, resolution: str = "") -> str:
    return _sha256(prompt, model, str(duration), resolution)


def cache_get(key: str) -> Optional[str]:
    for ext in (".mp4", ".webm"):
        p = _CACHE_DIR / f"{key}{ext}"
        if p.exists():
            return str(p)
    return None


def cache_put(key: str, src_path: str) -> str:
    ext = Path(src_path).suffix or ".mp4"
    dst = _CACHE_DIR / f"{key}{ext}"
    if not dst.exists():
        dst.write_bytes(Path(src_path).read_bytes())
    return str(dst)


# ─── Job Persistence ────────────────────────────────────────────────────────


def _job_path(job_id: str) -> Path:
    return _JOBS_DIR / f"{job_id}.json"


def _save_job(job: dict):
    with _lock:
        _job_path(job["id"]).write_text(json.dumps(job, indent=2, default=str))


def _load_job(job_id: str) -> Optional[dict]:
    p = _job_path(job_id)
    if p.exists():
        return json.loads(p.read_text())
    return None


def list_jobs() -> list[dict]:
    jobs = []
    for p in sorted(_JOBS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        jobs.append(json.loads(p.read_text()))
    return jobs


def list_incomplete_jobs() -> list[dict]:
    return [j for j in list_jobs() if j.get("status") in ("queued", "running")]


# ─── Job Runner (background thread) ─────────────────────────────────────────


def start_job(
    job_id: str,
    title: str,
    fn: Callable[[Callable], dict],
    on_done: Optional[Callable] = None,
) -> str:
    job = {
        "id": job_id,
        "title": title,
        "status": "queued",
        "progress": 0.0,
        "message": "Queued...",
        "video_path": "",
        "error": "",
        "created_at": _now_iso(),
        "started_at": "",
        "finished_at": "",
    }
    _save_job(job)
    _running_jobs[job_id] = job

    def _run():
        def _progress(pct: float, msg: str = ""):
            with _lock:
                if job_id in _running_jobs:
                    _running_jobs[job_id]["progress"] = pct
                    _running_jobs[job_id]["message"] = msg
                    _save_job(_running_jobs[job_id])

        with _lock:
            if job_id in _running_jobs:
                _running_jobs[job_id]["status"] = "running"
                _running_jobs[job_id]["started_at"] = _now_iso()
                _save_job(_running_jobs[job_id])

        try:
            result = fn(_progress)
            video_path = result.get("video_path", "") if isinstance(result, dict) else str(result)
            with _lock:
                if job_id in _running_jobs:
                    _running_jobs[job_id].update(
                        status="completed", progress=1.0, message="Complete!",
                        video_path=video_path, finished_at=_now_iso(),
                    )
                    _save_job(_running_jobs[job_id])
            if on_done:
                on_done(result)
        except Exception as e:
            with _lock:
                if job_id in _running_jobs:
                    _running_jobs[job_id].update(
                        status="failed", progress=1.0, message=str(e),
                        error=str(e), finished_at=_now_iso(),
                    )
                    _save_job(_running_jobs[job_id])
        finally:
            with _lock:
                _running_jobs.pop(job_id, None)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return job_id


def get_job(job_id: str) -> dict:
    with _lock:
        if job_id in _running_jobs:
            return dict(_running_jobs[job_id])
    j = _load_job(job_id)
    return j or {"id": job_id, "status": "unknown", "progress": 0, "message": "Not found"}


def get_all_running_jobs() -> list[dict]:
    with _lock:
        return [dict(j) for j in _running_jobs.values()]


# ─── Cleanup stale status files on exit ─────────────────────────────────────


def _cleanup():
    for jid in list(_running_jobs.keys()):
        with _lock:
            if jid in _running_jobs:
                _running_jobs[jid]["status"] = "interrupted"
                _running_jobs[jid]["message"] = "Process interrupted"
                _save_job(_running_jobs[jid])


atexit.register(_cleanup)
