# 音声文字起こしWebアプリ

## 課題
会議や打ち合わせの録音を手作業で文字起こしすると時間がかかり、話者の区別や要点整理まで手が回らない。

## 解決
音声ファイルをアップロードするだけで、文字起こし・話者分離・要約・タスク抽出までをワンストップで処理するWebアプリを構築。最大2GBのファイルをチャンク分割して処理でき、WebSocketでリアルタイムに進捗を表示する。

## 技術構成
- **Backend**: Python / FastAPI / asyncio
- **音声認識**: Groq Whisper API（大容量ファイルは自動チャンク分割、7分単位+3秒オーバーラップ）
- **話者分離**: AssemblyAI API（Groqへの自動フォールバック付き — カバー率50%未満で切替）
- **AI分析**: Anthropic Claude API（要約・主要トピック・決定事項の抽出、タスク一覧の生成）
- **リアルタイム通信**: WebSocket（進捗通知）+ ポーリングフォールバック
- **フロントエンド**: Jinja2テンプレート / Vanilla JS（ドラッグ&ドロップ、モーダルUI）
- **デプロイ**: Railway（Procfile / nixpacks 設定済み）

## 対応フォーマット
`.mp3` `.wav` `.m4a` `.mp4` `.flac` `.ogg` `.opus` `.aac` `.webm`

## 使い方

```bash
pip install -r requirements.txt
cp .env.example .env
# .env に各APIキーを設定（GROQ_API_KEY は必須、他は任意）
uvicorn app:app --reload
```

ブラウザで `http://localhost:8000` を開き、音声ファイルをドラッグ&ドロップ。

---
開発: Nekko Lab（Ryu / masanaritanaka）

