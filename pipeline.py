from datetime import datetime
from pathlib import Path
from typing import Callable

from config import ASSEMBLYAI_API_KEY, OUTPUT_DIR
from transcriber import convert_to_wav, transcribe
from diarizer import transcribe_with_diarization
from analyzer import analyze
from memory_utils import log_memory, release_memory


def _format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"[{h:02d}:{m:02d}:{s:02d}]"
    return f"[{m:02d}:{s:02d}]"


def _format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}時間{m}分{s}秒"
    return f"{m}分{s}秒"


def _build_transcript_text(segments: list[dict], options: dict) -> str:
    show_timestamps = options.get("timestamps", True)
    lines = []
    prev_speaker = None

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue

        speaker = seg.get("speaker", "")
        if speaker and speaker != prev_speaker:
            lines.append(f"\n【{speaker}】")
            prev_speaker = speaker

        if show_timestamps:
            lines.append(f"{_format_timestamp(seg['start'])} {text}")
        else:
            lines.append(text)

    return "\n".join(lines).strip()


def _build_output_file(
    filename: str,
    duration: float,
    num_speakers: int,
    transcript_text: str,
    analysis_text: str,
) -> str:
    sep = "=" * 48
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header_lines = [
        sep,
        "文字起こし結果",
        f"ファイル名: {filename}",
        f"処理日時: {now}",
        f"音声長さ: {_format_duration(duration)}",
    ]
    if num_speakers > 0:
        header_lines.append(f"話者数: {num_speakers}名")
    header_lines.append(sep)

    parts = ["\n".join(header_lines), "", "## 文字起こし", "", transcript_text]

    if analysis_text:
        parts += ["", sep, analysis_text]

    return "\n".join(parts)


async def process_job(
    job_id: str,
    file_path: Path,
    options: dict,
    callback: Callable,
) -> Path:
    wav_path = file_path.parent / f"{job_id}_audio.wav"
    transcript_path = OUTPUT_DIR / f"{job_id}_transcript.txt"
    want_diarize = options.get("diarize", True)
    use_assemblyai = want_diarize and bool(ASSEMBLYAI_API_KEY)

    try:
        # Step 1: WAV変換（durationも同時取得、AudioSegment不使用）
        await callback({"type": "progress", "stage": "converting", "percent": 3,
                        "message": "音声ファイルを変換中..."})
        log_memory("before_convert")
        duration = convert_to_wav(file_path, wav_path)
        file_path.unlink(missing_ok=True)
        log_memory("after_convert")

        # Step 2 & 3: 文字起こし（+ 話者分離）
        if use_assemblyai:
            segments = await transcribe_with_diarization(wav_path, job_id, callback)

            covered = max((s["end"] for s in segments), default=0)
            if covered < duration * 0.5:
                await callback({
                    "type": "progress",
                    "stage": "transcribing",
                    "percent": 22,
                    "message": f"AssemblyAI の日本語処理が不完全（{covered:.0f}秒/{duration:.0f}秒）→ Groq で全文字起こしに切り替えます",
                })
                segments = await transcribe(wav_path, job_id, callback)
                num_speakers = 0
            else:
                num_speakers = len(set(s.get("speaker", "") for s in segments if s.get("speaker")))
        else:
            await callback({"type": "progress", "stage": "transcribing", "percent": 20,
                            "message": "文字起こしを開始中..."})
            segments = await transcribe(wav_path, job_id, callback)
            num_speakers = 0

        log_memory("after_transcription")

        # Step 4: テキスト整形
        await callback({"type": "progress", "stage": "formatting", "percent": 82,
                        "message": "テキストを整形中..."})
        transcript_text = _build_transcript_text(segments, options)
        del segments

        transcript_path.write_text(transcript_text, encoding="utf-8")

        # Step 5: Claude分析（オプション）
        analysis_text = ""
        if options.get("summarize") or options.get("extract_tasks"):
            await callback({"type": "progress", "stage": "analyzing", "percent": 88,
                            "message": "Claude APIで分析中..."})
            analysis_text = await analyze(
                transcript_text,
                summarize=options.get("summarize", False),
                extract_tasks=options.get("extract_tasks", False),
            )

        # Step 6: ファイル出力
        await callback({"type": "progress", "stage": "saving", "percent": 96,
                        "message": "テキストファイルを保存中..."})
        output_text = _build_output_file(
            filename=options.get("original_filename", file_path.name),
            duration=duration,
            num_speakers=num_speakers,
            transcript_text=transcript_text,
            analysis_text=analysis_text,
        )
        output_path = OUTPUT_DIR / f"{job_id}.txt"
        output_path.write_text(output_text, encoding="utf-8")
        del transcript_text, analysis_text, output_text

        await callback({
            "type": "complete",
            "stage": "complete",
            "percent": 100,
            "message": "処理完了",
            "download_url": f"/download/{job_id}",
        })

        release_memory()
        log_memory("after_job_complete")
        return output_path

    except Exception:
        transcript_path.unlink(missing_ok=True)
        (OUTPUT_DIR / f"{job_id}.txt").unlink(missing_ok=True)
        raise
    finally:
        wav_path.unlink(missing_ok=True)
        file_path.unlink(missing_ok=True)
        # チャンクファイルが残っていれば削除
        for chunk_file in wav_path.parent.glob(f"{job_id}_chunk_*.wav"):
            chunk_file.unlink(missing_ok=True)
        release_memory()
        log_memory("after_job_cleanup")
