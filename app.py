import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel

from config import (
    ANTHROPIC_API_KEY,
    ASSEMBLYAI_API_KEY,
    FILE_RETENTION_HOURS,
    GROQ_API_KEY,
    GROQ_WHISPER_MODEL,
    MAX_FILE_SIZE_BYTES,
    OUTPUT_DIR,
    SUPPORTED_FORMATS,
    TEMPLATE_DIR,
    UPLOAD_DIR,
)
from analyzer import analyze
from pipeline import process_job
from memory_utils import log_memory, release_memory

logger = logging.getLogger(__name__)

JOB_RETENTION_SECONDS = 3600


# ---- ジョブ管理 ----

@dataclass
class Job:
    job_id: str
    filename: str
    status: str = "pending"
    percent: int = 0
    stage: str = ""
    message: str = ""
    output_path: Optional[Path] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

jobs: dict[str, Job] = {}
ws_connections: dict[str, list[WebSocket]] = {}


def cleanup_old_files():
    cutoff = datetime.now() - timedelta(hours=FILE_RETENTION_HOURS)
    for directory in [UPLOAD_DIR, OUTPUT_DIR]:
        for f in directory.iterdir():
            try:
                if datetime.fromtimestamp(f.stat().st_mtime) < cutoff:
                    f.unlink()
            except Exception:
                pass


def _cleanup_jobs():
    now = datetime.now()
    expired = [
        jid for jid, job in jobs.items()
        if job.status in ("complete", "error")
        and job.completed_at
        and (now - job.completed_at).total_seconds() > JOB_RETENTION_SECONDS
    ]
    for jid in expired:
        job = jobs.pop(jid, None)
        if job and job.output_path:
            job.output_path.unlink(missing_ok=True)
        transcript = OUTPUT_DIR / f"{jid}_transcript.txt"
        transcript.unlink(missing_ok=True)
        ws_connections.pop(jid, None)

    orphaned = [jid for jid in ws_connections if jid not in jobs]
    for jid in orphaned:
        ws_connections.pop(jid, None)

    if expired or orphaned:
        logger.info(f"[cleanup] jobs={len(expired)} ws_orphans={len(orphaned)} remaining_jobs={len(jobs)}")


async def _periodic_cleanup():
    while True:
        await asyncio.sleep(300)
        _cleanup_jobs()
        cleanup_old_files()
        release_memory()
        log_memory("periodic_cleanup")


# ---- FastAPI アプリ ----

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cleanup_old_files()
    log_memory("startup")
    task = asyncio.create_task(_periodic_cleanup())
    yield
    task.cancel()

app = FastAPI(title="文字起こしツール", lifespan=lifespan)
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


# ---- ヘルパー ----

async def broadcast(job_id: str, msg: dict):
    job = jobs.get(job_id)
    if job:
        job.percent = msg.get("percent", job.percent)
        job.stage = msg.get("stage", job.stage)
        job.message = msg.get("message", job.message)
        if msg.get("type") == "complete":
            job.status = "complete"
            job.completed_at = datetime.now()
        elif msg.get("type") == "error":
            job.status = "error"
            job.error = msg.get("message", "エラーが発生しました")
            job.completed_at = datetime.now()
        elif msg.get("type") == "progress":
            job.status = "processing"

    text = json.dumps(msg, ensure_ascii=False)
    dead = []
    for ws in ws_connections.get(job_id, []):
        try:
            await ws.send_text(text)
        except Exception:
            dead.append(ws)
    for ws in dead:
        ws_connections[job_id].remove(ws)


# ---- ルート ----

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "whisper_model": GROQ_WHISPER_MODEL,
        "groq_available": bool(GROQ_API_KEY),
        "claude_available": bool(ANTHROPIC_API_KEY),
        "diarization_available": bool(ASSEMBLYAI_API_KEY),
    }


@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    options: str = Form("{}"),
):
    suffix = Path(file.filename or "audio").suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=400, detail=f"非対応のファイル形式です: {suffix}")

    job_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{job_id}{suffix}"

    size = 0
    async with aiofiles.open(save_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_FILE_SIZE_BYTES:
                await f.close()
                save_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="ファイルサイズが大きすぎます（最大2GB）")
            await f.write(chunk)

    try:
        opts = json.loads(options)
    except Exception:
        opts = {}

    job = Job(job_id=job_id, filename=file.filename or "audio")
    jobs[job_id] = job
    ws_connections[job_id] = []

    asyncio.create_task(
        _run_job(job_id, save_path, opts)
    )

    return {"job_id": job_id, "filename": file.filename}


async def _run_job(job_id: str, file_path: Path, options: dict):
    async def cb(msg: dict):
        await broadcast(job_id, msg)

    try:
        output_path = await process_job(job_id, file_path, options, cb)
        jobs[job_id].output_path = output_path
    except Exception as e:
        await broadcast(job_id, {
            "type": "error",
            "stage": "error",
            "percent": 0,
            "message": f"処理中にエラーが発生しました: {e}",
        })


@app.get("/status/{job_id}")
async def status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません")
    return {
        "job_id": job.job_id,
        "status": job.status,
        "percent": job.percent,
        "stage": job.stage,
        "message": job.message,
        "error": job.error,
        "download_url": f"/download/{job_id}" if job.status == "complete" else None,
    }


@app.get("/download/{job_id}")
async def download(job_id: str):
    job = jobs.get(job_id)
    if not job or job.status != "complete" or not job.output_path:
        raise HTTPException(status_code=404, detail="ファイルが見つかりません")
    if not job.output_path.exists():
        raise HTTPException(status_code=404, detail="ファイルが見つかりません")

    safe_name = Path(job.filename).stem + "_文字起こし.txt"
    return FileResponse(
        path=job.output_path,
        filename=safe_name,
        media_type="text/plain; charset=utf-8",
    )


class AnalyzeRequest(BaseModel):
    summarize: bool = False
    extract_tasks: bool = False


@app.get("/text/{job_id}")
async def get_text(job_id: str):
    output_path = OUTPUT_DIR / f"{job_id}.txt"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="ファイルが見つかりません")
    return {"text": output_path.read_text(encoding="utf-8")}


@app.post("/analyze/{job_id}")
async def run_analysis(job_id: str, body: AnalyzeRequest):
    output_path = OUTPUT_DIR / f"{job_id}.txt"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="ジョブが見つかりません。再度アップロードしてください。")

    transcript_path = OUTPUT_DIR / f"{job_id}_transcript.txt"
    if transcript_path.exists():
        transcript_text = transcript_path.read_text(encoding="utf-8")
    else:
        # 旧バージョン処理: 出力ファイルから文字起こし部分を抽出
        content = output_path.read_text(encoding="utf-8")
        marker = "## 文字起こし\n\n"
        pos = content.find(marker)
        if pos == -1:
            raise HTTPException(status_code=404, detail="文字起こしデータが見つかりません")
        start = pos + len(marker)
        sep_pos = content.find("\n\n" + "=" * 48 + "\n", start)
        transcript_text = content[start:sep_pos].strip() if sep_pos != -1 else content[start:].strip()

    analysis_text = await analyze(
        transcript_text,
        summarize=body.summarize,
        extract_tasks=body.extract_tasks,
    )

    # 既存の出力ファイルにAI分析を追記（既存の分析があれば置き換え）
    sep = "=" * 48
    content = output_path.read_text(encoding="utf-8")
    marker = "\n\n" + sep + "\n"
    transcript_pos = content.find("## 文字起こし")
    if transcript_pos != -1:
        analysis_sep_pos = content.find(marker, transcript_pos)
        base = content[:analysis_sep_pos] if analysis_sep_pos != -1 else content
    else:
        base = content
    updated = base + marker + analysis_text if analysis_text else base
    output_path.write_text(updated, encoding="utf-8")

    job = jobs.get(job_id)
    if job:
        job.output_path = output_path

    return {"analysis": analysis_text, "full_text": updated}


@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await websocket.accept()

    if job_id not in ws_connections:
        ws_connections[job_id] = []
    ws_connections[job_id].append(websocket)

    # 既に完了・エラーの場合は即通知
    job = jobs.get(job_id)
    if job:
        if job.status == "complete":
            await websocket.send_text(json.dumps({
                "type": "complete",
                "stage": "complete",
                "percent": 100,
                "message": "処理完了",
                "download_url": f"/download/{job_id}",
            }, ensure_ascii=False))
        elif job.status == "error":
            await websocket.send_text(json.dumps({
                "type": "error",
                "stage": "error",
                "percent": 0,
                "message": job.error or "エラーが発生しました",
            }, ensure_ascii=False))
        elif job.status in ("processing", "pending"):
            await websocket.send_text(json.dumps({
                "type": "progress",
                "stage": job.stage,
                "percent": job.percent,
                "message": job.message,
            }, ensure_ascii=False))

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if job_id in ws_connections and websocket in ws_connections[job_id]:
            ws_connections[job_id].remove(websocket)
