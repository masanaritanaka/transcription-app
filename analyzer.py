import anthropic

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL

_client = None


def get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY が設定されていません")
        _client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    return _client


_SUMMARIZE_PROMPT = """\
以下は音声の文字起こしテキストです。日本語で次の3点をまとめてください。

1. **要約**（300字以内）
2. **主要なトピック**（箇条書き）
3. **重要な決定事項**（あれば箇条書き、なければ「なし」）

文字起こし:
{transcript}"""

_TASKS_PROMPT = """\
以下の文字起こしからタスク・アクションアイテムを抽出してください。
形式: 「- [ ] [担当者] タスク内容」
担当者が不明な場合は担当者欄を省略してください。
タスクがなければ「なし」と記載してください。

文字起こし:
{transcript}"""


async def analyze(
    transcript: str,
    summarize: bool,
    extract_tasks: bool,
) -> str:
    if not ANTHROPIC_API_KEY:
        return "※ ANTHROPIC_API_KEY が未設定のため分析をスキップしました"

    # Claude Sonnet 4.6 は200Kコンテキスト対応
    truncated = transcript[:150000]
    results = []

    try:
        client = get_client()

        if summarize:
            response = await client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": _SUMMARIZE_PROMPT.format(transcript=truncated),
                }],
            )
            results.append("## 要約\n\n" + response.content[0].text)

        if extract_tasks:
            response = await client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1500,
                messages=[{
                    "role": "user",
                    "content": _TASKS_PROMPT.format(transcript=truncated),
                }],
            )
            results.append("## タスク一覧\n\n" + response.content[0].text)

    except anthropic.AuthenticationError:
        return "※ Claude API認証エラー: APIキーを確認してください"
    except anthropic.RateLimitError:
        return "※ Claude APIのレート制限に達しました。しばらく後に再試行してください"
    except Exception as e:
        return f"※ Claude API エラー: {e}"

    return "\n\n".join(results)
