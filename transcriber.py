import asyncio
import io
import os
from pathlib import Path
from typing import Callable

from groq import Groq
from pydub import AudioSegment

from config import (
    CHUNK_DURATION_SECONDS,
    CHUNK_OVERLAP_SECONDS,
    GROQ_API_KEY,
    GROQ_MAX_FILE_BYTES,
    GROQ_WHISPER_MODEL,
    UPLOAD_DIR,
)

_client = None


def get_client() -> Groq:
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY が設定されていません")
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def convert_to_wav(input_path: Path, output_path: Path) -> Path:
    audio = AudioSegment.from_file(str(input_path))
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(str(output_path), format="wav")
    return output_path


def get_audio_duration(wav_path: Path) -> float:
    audio = AudioSegment.from_wav(str(wav_path))
    return len(audio) / 1000.0


def _needs_chunking(wav_path: Path) -> bool:
    return wav_path.stat().st_size > GROQ_MAX_FILE_BYTES


def _chunk_audio(wav_path: Path) -> list[tuple[float, AudioSegment]]:
    audio = AudioSegment.from_wav(str(wav_path))
    total_ms = len(audio)
    chunk_ms = CHUNK_DURATION_SECONDS * 1000
    overlap_ms = CHUNK_OVERLAP_SECONDS * 1000

    chunks = []
    start_ms = 0
    while start_ms < total_ms:
        end_ms = min(start_ms + chunk_ms, total_ms)
        chunks.append((start_ms / 1000.0, audio[start_ms:end_ms]))
        if end_ms == total_ms:
            break
        start_ms += chunk_ms - overlap_ms
    return chunks


def _transcribe_segment(audio_bytes: bytes, filename: str, language: str = "ja") -> list[dict]:
    client = get_client()
    # timestamp_granularities は省略（指定するとレスポンスが途中で切れる場合がある）
    response = client.audio.transcriptions.create(
        file=(filename, audio_bytes, "audio/wav"),
        model=GROQ_WHISPER_MODEL,
        language=language,
        response_format="verbose_json",
    )
    segments = []
    for seg in (response.segments or []):
        # Groq SDK は dict または属性アクセス両方ありうるため両対応
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
        # ファイルが小さい場合は1回で送信
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
        return segments

    # 長時間ファイルはチャンク分割
    chunks = await loop.run_in_executor(None, lambda: _chunk_audio(wav_path))
    total_chunks = len(chunks)
    all_segments: list[dict] = []

    for i, (start_sec, chunk) in enumerate(chunks):
        base_percent = 22 + int((i / total_chunks) * 58)
        await callback({
            "type": "progress",
            "stage": "transcribing",
            "percent": base_percent,
            "message": f"文字起こし中... (チャンク {i + 1}/{total_chunks})",
            "chunk_index": i + 1,
            "total_chunks": total_chunks,
        })

        try:
            buf = io.BytesIO()
            chunk.export(buf, format="wav")
            audio_bytes = buf.getvalue()

            raw_segs = await loop.run_in_executor(
                None,
                lambda b=audio_bytes: _transcribe_segment(b, f"chunk_{i}.wav", language),
            )

            half_overlap = CHUNK_OVERLAP_SECONDS / 2.0
            for seg in raw_segs:
                if i > 0 and seg["start"] < half_overlap:
                    continue
                all_segments.append({
                    "start": seg["start"] + start_sec,
                    "end": seg["end"] + start_sec,
                    "text": seg["text"],
                    "words": [],
                })

        except Exception as e:
            all_segments.append({
                "start": start_sec,
                "end": start_sec,
                "text": f"[チャンク {i + 1} 処理エラー: {e}]",
                "words": [],
            })

    return all_segments
