import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
TEMPLATE_DIR = BASE_DIR / "templates"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Groq Whisper API
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_WHISPER_MODEL = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3")

# Groq の1リクエストあたりの最大ファイルサイズ（25MB）
GROQ_MAX_FILE_MB = 24
GROQ_MAX_FILE_BYTES = GROQ_MAX_FILE_MB * 1024 * 1024

# チャンク設定（25MB制限に合わせて7分）
CHUNK_DURATION_SECONDS = 420   # 7分
CHUNK_OVERLAP_SECONDS = 3      # 3秒オーバーラップ

# AssemblyAI API（話者分離 + 文字起こし）
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")
ASSEMBLYAI_BASE_URL = "https://api.assemblyai.com/v2"

# Claude API
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-6"

# ファイル保持時間
FILE_RETENTION_HOURS = 24

# 最大ファイルサイズ（2GB）
MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024 * 1024

# 対応フォーマット
SUPPORTED_FORMATS = {".mp3", ".wav", ".m4a", ".mp4", ".flac", ".ogg",
                     ".opus", ".aac", ".weba", ".webm"}
