# 🎙️ 音声文字起こしWebアプリ

Groq Whisper + AssemblyAI + Claude APIを組み合わせた、日本語対応の音声文字起こしWebアプリです。

## 機能

- 音声・動画ファイルのアップロードと自動文字起こし（Groq Whisper）
- 話者分離（AssemblyAI）
- WebSocketによるリアルタイム進捗表示
- Claude APIによる要約・タスク抽出
- 大容量ファイルのチャンク分割処理（最大2GB）
- 結果テキストのダウンロード

## 対応フォーマット

`.mp3` `.wav` `.m4a` `.mp4` `.flac` `.ogg` `.opus` `.aac` `.webm`

## 技術スタック

| カテゴリ | 使用技術 |
|---|---|
| Backend | Python / FastAPI / asyncio |
| 音声認識 | Groq Whisper API |
| 話者分離 | AssemblyAI API |
| AI分析 | Anthropic Claude API |
| リアルタイム通信 | WebSocket |
| デプロイ | Railway |

## セットアップ

```bash
pip install -r requirements.txt
cp .env.example .env
# .envにAPIキーを設定
uvicorn app:app --reload
```

## 環境変数
GROQ_API_KEY=
ASSEMBLYAI_API_KEY=
ANTHROPIC_API_KEY=
## 開発：Nekko Lab（Ryu）
