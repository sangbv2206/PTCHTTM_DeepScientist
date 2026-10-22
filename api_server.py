import asyncio
import json
import os
import threading
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pipeline.orchestrator import DeepScientistPipeline


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
WEB_DIR = ROOT / "web"


def _load_env_file() -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


_load_env_file()

app = FastAPI(title="DeepScientist Research Workspace", version="0.1.0")
app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")

_runs: Dict[str, Dict[str, Any]] = {}
_events: Dict[str, deque] = {}
_lock = threading.Lock()


def _load_existing_runs() -> None:
    if not OUTPUT_DIR.exists():
        return
    for run_dir in OUTPUT_DIR.glob("run_*"):
        status_file = run_dir / "status.json"
        if not status_file.is_file():
            continue
        try:
            saved = json.loads(status_file.read_text(encoding="utf-8"))
            run_id = saved.get("run_id") or run_dir.name.removeprefix("run_")
            saved["run_id"] = run_id
            saved["run_dir"] = str(run_dir)
            saved["has_pdf"] = (run_dir / "latex" / "paper.pdf").is_file()
            # Dọn dẹp Zombie Runs nếu máy chủ bị khởi động lại giữa chừng
            if saved.get("status") in {"running", "queued"}:
                saved["status"] = "interrupted"
                saved["message"] = "Bị gián đoạn do máy chủ khởi động lại"
                try:
                    status_file.write_text(json.dumps(saved, indent=2, ensure_ascii=False), encoding="utf-8")
                except Exception:
                    pass
            _runs[run_id] = saved
            _events[run_id] = deque(maxlen=200)
        except (OSError, json.JSONDecodeError):
            continue


_load_existing_runs()


DEFAULT_MODEL = os.getenv("GEMINI_MODEL") or "gemini-2.0-flash"

class RunRequest(BaseModel):
    topic: str = Field(min_length=3, max_length=500)
    model: Optional[str] = Field(default=None)
    template: Optional[str] = None
    iterations: int = Field(default=2, ge=1, le=5)
    offline: bool = False


def _save_status(run_id: str) -> None:
    run = _runs[run_id]
    run_dir = run.get("run_dir")
    if not run_dir:
        return
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    status = {key: value for key, value in run.items() if key != "result"}
    (Path(run_dir) / "status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")


def _public_run(run: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in run.items() if key not in {"run_dir", "result"}}


def _public_event(event: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in event.items() if key not in {"run_dir", "result"}}


def _on_event(event: Dict[str, Any]) -> None:
    run_id = event["run_id"]
    with _lock:
        run = _runs.get(run_id)
        if not run:
            return
        run.update({key: value for key, value in event.items() if key != "result"})
        if event.get("run_dir"):
            run["run_dir"] = event["run_dir"]
        if event.get("status") == "completed":
            run["status"] = "completed"
            result = event.get("result") or {}
            run["has_pdf"] = bool(result.get("pdf_path")) and Path(result["pdf_path"]).is_file()
        _events.setdefault(run_id, deque(maxlen=200)).append(_public_event(event))
        _save_status(run_id)


def _run_pipeline(run_id: str, request: RunRequest) -> None:
    try:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not configured on the server")
        model_to_use = request.model or DEFAULT_MODEL
        pipeline = DeepScientistPipeline(
            api_key=api_key,
            model_name=model_to_use,
            output_base_dir=str(OUTPUT_DIR),
            online_search=not request.offline,
            semantic_scholar_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
            event_callback=_on_event,
        )
        result = pipeline.run(
            topic=request.topic,
            template_name=request.template,
            max_iterations=request.iterations,
            run_id=run_id,
        )
    except Exception as error:
        with _lock:
            run = _runs.get(run_id)
            if run:
                run.update({"status": "failed", "error": str(error), "progress": run.get("progress", 0.0)})
                _events.setdefault(run_id, deque(maxlen=200)).append(_public_event({
                    "run_id": run_id,
                    "stage": run.get("stage", 0),
                    "status": "failed",
                    "message": str(error),
                    "progress": run.get("progress", 0.0),
                }))
                _save_status(run_id)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "gemini_configured": str(bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))).lower(),
        "default_model": DEFAULT_MODEL,
    }


@app.get("/api/templates")
def templates() -> Dict[str, Any]:
    template_root = ROOT / "templates"
    names = sorted(path.name for path in template_root.iterdir() if path.is_dir() and (path / "experiment.py").exists())
    return {"templates": names}


@app.post("/api/runs", status_code=202)
def create_run(request: RunRequest) -> Dict[str, Any]:
    run_id = uuid.uuid4().hex[:12]
    with _lock:
        _runs[run_id] = {
            "run_id": run_id,
            "topic": request.topic,
            "status": "queued",
            "stage": 0,
            "progress": 0.0,
            "message": "Queued for execution",
        }
        _events[run_id] = deque(maxlen=200)
    threading.Thread(target=_run_pipeline, args=(run_id, request), daemon=True).start()
    return _public_run(_runs[run_id])


@app.post("/api/runs/{run_id}/rerun", status_code=202)
def rerun_run(run_id: str) -> Dict[str, Any]:
    with _lock:
        old_run = _runs.get(run_id)
        if not old_run:
            raise HTTPException(status_code=404, detail="Run not found")
        if old_run.get("status") in {"running", "queued"}:
            raise HTTPException(status_code=409, detail="Lượt nghiên cứu này đang trong quá trình thực hiện!")

        topic = old_run.get("topic", "")
        if not topic:
            raise HTTPException(status_code=400, detail="Cannot rerun: missing topic")
        req = RunRequest(
            topic=topic,
            model=old_run.get("model") or DEFAULT_MODEL,
            template=old_run.get("selected_template"),
            iterations=old_run.get("iterations", 2),
            offline=old_run.get("offline", False)
        )
        # Chạy lại TRỰC TIẾP trên cùng run_id (In-place Rerun), không sinh card mới làm trùng lặp
        old_run.update({
            "status": "queued",
            "stage": 0,
            "progress": 0.0,
            "message": "Đang khởi chạy lại…",
            "has_pdf": False,
        })
        _events[run_id] = deque(maxlen=200)
        _save_status(run_id)

    threading.Thread(target=_run_pipeline, args=(run_id, req), daemon=True).start()
    return _public_run(_runs[run_id])


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str) -> Dict[str, Any]:
    with _lock:
        run = _runs.get(run_id)
        run_dir = run.get("run_dir") if run else None
        if not run_dir:
            possible_dir = OUTPUT_DIR / f"run_{run_id}"
            if possible_dir.exists():
                run_dir = str(possible_dir)

        # Đánh dấu ngắt nếu run đang chạy dở
        if run:
            run["status"] = "interrupted"
            run["message"] = "Đã bị người dùng hủy và xóa"

        if run_dir and Path(run_dir).exists():
            import shutil
            import stat

            def _handle_remove_readonly(func, path, exc_info):
                try:
                    os.chmod(path, stat.S_IWRITE)
                    func(path)
                except Exception:
                    pass

            try:
                shutil.rmtree(run_dir, onerror=_handle_remove_readonly)
            except Exception as e:
                print(f"[DeleteRun] Cảnh báo khi xóa thư mục {run_dir}: {e}")

        _runs.pop(run_id, None)
        _events.pop(run_id, None)

    return {"success": True, "deleted_run_id": run_id}


@app.get("/api/runs")
def list_runs() -> Dict[str, Any]:
    with _lock:
        return {"runs": [_public_run(run) for run in reversed(list(_runs.values()))]}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> Dict[str, Any]:
    with _lock:
        run = _runs.get(run_id)
        if run:
            return _public_run(run)
    raise HTTPException(status_code=404, detail="Run not found")


@app.get("/api/runs/{run_id}/events")
async def stream_events(run_id: str) -> StreamingResponse:
    with _lock:
        if run_id not in _runs:
            raise HTTPException(status_code=404, detail="Run not found")

    async def event_stream():
        sent = 0
        while True:
            with _lock:
                if run_id not in _runs:
                    break
                events = list(_events.get(run_id, []))
                status = _runs[run_id].get("status", "failed")
            for event in events[sent:]:
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            sent = len(events)
            if status in {"completed", "failed", "interrupted"} and sent >= len(events):
                break
            try:
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                break

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/api/runs/{run_id}/artifacts")
def artifacts(run_id: str) -> Dict[str, Any]:
    with _lock:
        run = _runs.get(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        run_dir = run.get("run_dir")
    if not run_dir or not Path(run_dir).exists():
        return {"artifacts": []}
    root = Path(run_dir)
    files = [{"path": str(path.relative_to(root)).replace("\\", "/"), "size": path.stat().st_size} for path in root.rglob("*") if path.is_file()]
    return {"artifacts": sorted(files, key=lambda item: item["path"])}


@app.get("/api/runs/{run_id}/pdf")
def paper_pdf(run_id: str) -> FileResponse:
    with _lock:
        run = _runs.get(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        run_dir = run.get("run_dir")
    pdf_path = Path(run_dir) / "latex" / "paper.pdf" if run_dir else None
    if not pdf_path or not pdf_path.is_file():
        raise HTTPException(status_code=404, detail="Paper PDF is not ready")
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename="paper.pdf",
        content_disposition_type="inline",
    )


@app.get("/api/runs/{run_id}/artifacts/{artifact_path:path}")
def artifact(run_id: str, artifact_path: str) -> FileResponse:
    with _lock:
        run = _runs.get(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        run_dir = run.get("run_dir", "")
    root = Path(run_dir).resolve()
    target = (root / artifact_path).resolve()
    if root not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(target)
