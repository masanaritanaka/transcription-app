"""
AssemblyAI を使った文字起こし + 話者分離の統合モジュール。
話者分離ONのときはこのモジュールがGroqの代わりに文字起こしも担う。
"""
import asyncio
from pathlib import Path
from typing import Callable

import httpx

from config import ASSEMBLYAI_API_KEY, ASSEMBLYAI_BASE_URL

_POLL_INTERVAL = 5   # 秒
_MAX_WAIT = 3600     # 最大1時間


async def transcribe_with_diarization(
    wav_path: Path,
    job_id: str,
    callback: Callable,
    language: str = "ja",
) -> list[dict]:
    """
    AssemblyAI で文字起こし + 話者分離を同時実行。
    戻り値: [{"start": float, "end": float, "text": str, "speaker": str}]
    """
    json_headers = {
        "authorization": ASSEMBLYAI_API_KEY,
        "content-type": "application/json",
    }

    # アップロードは大きなファイルも考慮して長めのタイムアウト
    async with httpx.AsyncClient(timeout=600.0) as client:

        # Step 1: 音声ファイルをアップロード
        await callback({
            "type": "progress",
            "stage": "diarizing",
            "percent": 8,
            "message": "音声ファイルをアップロード中...",
        })
        with open(wav_path, "rb") as f:
            upload_resp = await client.post(
                f"{ASSEMBLYAI_BASE_URL}/upload",
                headers={"authorization": ASSEMBLYAI_API_KEY},
                content=f.read(),
            )
        upload_resp.raise_for_status()
        audio_url = upload_resp.json()["upload_url"]

        # Step 2: 文字起こし + 話者分離ジョブを投入
        # ※ speaker_labels=True のとき language_code は使えないため
        #   language_detection=True で自動検出する
        await callback({
            "type": "progress",
            "stage": "diarizing",
            "percent": 12,
            "message": "話者分離付き文字起こしを開始中...",
        })
        # speech_models は必須（universal-2 が日本語+話者分離対応）
        # まず speaker_labels あり → なし の順で試みる
        for attempt, payload in enumerate([
            {
                "audio_url": audio_url,
                "speech_models": "universal-2",
                "language_code": "ja",
                "speaker_labels": True,
            },
            {
                "audio_url": audio_url,
                "speech_models": "universal-2",
                "language_code": "ja",
            },
        ]):
            transcript_resp = await client.post(
                f"{ASSEMBLYAI_BASE_URL}/transcript",
                headers=json_headers,
                json=payload,
            )
            if transcript_resp.is_success:
                break
            if attempt == 0:
                await callback({
                    "type": "progress",
                    "stage": "diarizing",
                    "percent": 13,
                    "message": f"話者分離を外して再試行中... ({transcript_resp.status_code})",
                })
                continue
            raise RuntimeError(
                f"AssemblyAI エラー ({transcript_resp.status_code}): {transcript_resp.text}"
            )
        transcript_id = transcript_resp.json()["id"]

        # Step 3: 完了までポーリング
        waited = 0
        last_percent = 12
        while waited < _MAX_WAIT:
            await asyncio.sleep(_POLL_INTERVAL)
            waited += _POLL_INTERVAL

            status_resp = await client.get(
                f"{ASSEMBLYAI_BASE_URL}/transcript/{transcript_id}",
                headers={"authorization": ASSEMBLYAI_API_KEY},
            )
            status_resp.raise_for_status()
            data = status_resp.json()
            status = data.get("status", "")

            # 進捗を少しずつ進める演出（78%上限を撤廃し経過時間を表示）
            if last_percent < 78:
                last_percent = min(last_percent + 3, 78)
            elapsed_min = int(waited // 60)
            elapsed_sec = int(waited % 60)
            elapsed_str = f"{elapsed_min}分{elapsed_sec}秒" if elapsed_min > 0 else f"{elapsed_sec}秒"
            await callback({
                "type": "progress",
                "stage": "transcribing",
                "percent": last_percent,
                "message": f"AssemblyAI で処理中... ({elapsed_str}経過)",
            })

            if status == "completed":
                segments = []
                utterances = data.get("utterances") or []
                if utterances:
                    # speaker_labels あり
                    for u in utterances:
                        segments.append({
                            "start": u["start"] / 1000.0,
                            "end":   u["end"]   / 1000.0,
                            "text":  u["text"].strip(),
                            "speaker": f"SPEAKER_{u['speaker']}",
                            "words": [],
                        })
                else:
                    # speaker_labels なし → words / sentences を使う
                    words = data.get("words") or []
                    full_text = data.get("text") or ""
                    if words:
                        # word単位でセグメント化（句読点で区切る）
                        buf_text, buf_start, buf_end = [], None, None
                        for w in words:
                            if buf_start is None:
                                buf_start = w["start"] / 1000.0
                            buf_end = w["end"] / 1000.0
                            buf_text.append(w["text"])
                            if w["text"].endswith(("。", "、", "？", "！", ".", ",", "?", "!")):
                                segments.append({
                                    "start": buf_start,
                                    "end": buf_end,
                                    "text": "".join(buf_text).strip(),
                                    "speaker": "",
                                    "words": [],
                                })
                                buf_text, buf_start, buf_end = [], None, None
                        if buf_text:
                            segments.append({
                                "start": buf_start,
                                "end": buf_end,
                                "text": "".join(buf_text).strip(),
                                "speaker": "",
                                "words": [],
                            })
                    elif full_text:
                        segments.append({
                            "start": 0.0, "end": 0.0,
                            "text": full_text, "speaker": "", "words": [],
                        })
                num_speakers = len(set(s["speaker"] for s in segments if s["speaker"]))
                await callback({
                    "type": "progress",
                    "stage": "transcribing",
                    "percent": 80,
                    "message": f"文字起こし + 話者分離完了（{num_speakers}名）",
                })
                return segments

            elif status == "error":
                error_msg = data.get("error", "不明なエラー")
                raise RuntimeError(f"AssemblyAI エラー: {error_msg}")

    raise TimeoutError("AssemblyAI の処理がタイムアウトしました")
