import asyncio
import io
import json
import subprocess
from pathlib import Path
from typing import Callable

from groq import Groq

from config import (
    CHUNK_DURATION_SECONDS,
    CHUNK_OVERLAP_SECONDS,
    GROQ_API_KEY,
    GROQ_MAX_FILE_BYTES,
    GROQ_WHISPER_MODEL,
)
from memory_utils import log_memory, release_memory

_client = None


def get_client() -> Groq:
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY が設定されていません")
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def convert_to_wav(input_path: Path, output_path: Path) -> float:
    """ffmpegでWAV変換し、durationを返す。Pythonメモリに音声データを展開しない。"""
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(input_path),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            str(output_path),
        ],
        capture_output=True,
        check=True,
    )
    return get_audio_duration(output_path)


def get_audio_duration(file_path: Path) -> float:
    """ffprobeでdurationを取得。メモリに音声データを展開しない。"""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                str(file_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return float(data["format"]["duration"])
    except Exception:
        pass
    # フォールバック: 16kHz mono 16-bit PCM WAVのヘッダ計算
    try:
        size = file_path.stat().st_size
        return max(0.0, (size - 44) / 32000.0)
    except Exception:
        return 0.0


def _needs_chunking(wav_path: Path) -> bool:
    return wav_path.stat().st_size > GROQ_MAX_FILE_BYTES


def _extract_chunk(wav_path: Path, chunk_path: Path, start_s: float, duration_s: float):
    """ffmpegで指定区間を切り出し。Pythonメモリ消費ゼロ。"""
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(wav_path),
            "-ss", str(start_s), "-t", str(duration_s),
            "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(chunk_path),
        ],
        capture_output=True,
        check=True,
    )


def _transcribe_segment(audio_bytes: bytes, filename: str, language: str = "ja") -> list[dict]:
    client = get_client()
    response = client.audio.transcriptions.create(
        file=(filename, audio_bytes, "audio/wav"),
        model=GROQ_WHISPER_MODEL,
        language=language,
        response_format="verbose_json",
    )
    segments = []
    for seg in (response.segments or []):
        if isinstance(seg, dict):
            start = float(seg.get("start", 0))
            end   = float(seg.get("end", 0))
            text  = str(seg.get("text", "")).strip()
        else:
            start = float(seg.start)
            end   = float(seg.end)
            text  = seg.text.strip()
        if text:
            segments.append({"start": start, "end": end, "text": text, "words": []})
    return segments


async def transcribe(
    wav_path: Path,
    job_id: str,
    callback: Callable,
    language: str = "ja",
) -> list[dict]:
    loop = asyncio.get_event_loop()

    if not _needs_chunking(wav_path):
        await callback({
            "type": "progress",
            "stage": "transcribing",
            "percent": 30,
            "message": "文字起こし中...",
            "chunk_index": 1,
            "total_chunks": 1,
        })
        audio_bytes = wav_path.read_bytes()
        segments = await loop.run_in_executor(
            None,
            lambda: _transcribe_segment(audio_bytes, "audio.wav", language),
        )
        del audio_bytes
        release_memory()
        log_memory("after_transcribe_small")
        return segments

    # 長時間ファイル: ffmpegでチャンクをディスク上に切り出し、1つずつ処理
    log_memory("before_chunking")
    duration = get_audio_duration(wav_path)

    positions: list[tuple[float, float]] = []
    start = 0.0
    while start < duration:
        end = min(start + CHUNK_DURATION_SECONDS, duration)
        positions.append((start, end - start))
        if end >= duration:
            break
        start += CHUNK_DURATION_SECONDS - CHUNK_OVERLAP_SECONDS

    total_chunks = len(positions)
    all_segments: list[dict] = []

    for i, (start_s, chunk_dur) in enumerate(positions):
        base_percent = 22 + int((i / total_chunks) * 58)
        await callback({
            "type": "progress",
            "stage": "transcribing",
            "percent": base_percent,
            "message": f"文字起こし中... (チャンク {i + 1}/{total_chunks})",
            "chunk_index": i + 1,
            "total_chunks": total_chunks,
        })

        chunk_path = wav_path.parent / f"{job_id}_chunk_{i}.wav"
        try:
            await loop.run_in_executor(
                None,
                lambda s=start_s, d=chunk_dur: _extract_chunk(wav_path, chunk_path, s, d),
            )

            audio_bytes = chunk_path.read_bytes()
            chunk_path.unlink(missing_ok=True)

            raw_segs = await loop.run_in_executor(
                None,
                lambda b=audio_bytes: _transcribe_segment(b, f"chunk_{i}.wav", language),
            )
            del audio_bytes

            half_overlap = CHUNK_OVERLAP_SECONDS / 2.0
            for seg in raw_segs:
                if i > 0 and seg["start"] < half_overlap:
                    continue
                all_segments.append({
                    "start": seg["start"] + start_s,
                    "end": seg["end"] + start_s,
                    "text": seg["text"],
                    "words": [],
                })
            del raw_segs

        except Exception as e:
            chunk_path.unlink(missing_ok=True)
            all_segments.append({
                "start": start_s,
                "end": start_s,
                "text": f"[チャンク {i + 1} 処理エラー: {e}]",
                "words": [],
            })

    release_memory()
    log_memory("after_transcribe_chunked")
    return all_segments
