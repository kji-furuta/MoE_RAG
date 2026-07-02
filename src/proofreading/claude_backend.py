"""
LLMバックエンド（Claude API / Ollama Chat フォールバック）
校正指示をsystemロールで強制し、要約への逸脱を防止する
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

# システムプロンプト（土木報告書・校正特化）
PROOFREADING_SYSTEM_PROMPT = (
    "あなたは土木・建設分野の日本語報告書の校正専門家です。\n"
    "以下のルールを厳守してください：\n"
    "1. 誤字・脱字・表記ゆれ・助詞の誤り・句読点の不自然さのみを指摘すること\n"
    "2. 要約・言い換え・内容の追加・推測・論評は絶対に禁止\n"
    "3. 原文の意味や内容は一切変更しないこと\n"
    "4. 数式（例: 1.20 × 0.218 ＝ 3.139）は校正対象外。数値や演算子を修正提案しないこと\n"
    "5. 以下は正しい土木専門用語であり、誤字として指摘しないこと：\n"
    "   盤下、桝、集水桝、チッピング、グレーチング、ｺﾝｸﾘｰﾄ、型枠、配筋、\n"
    "   曲線半径、縦断勾配、横断勾配、片勾配、拡幅、建築限界、\n"
    "   ボックスカルバート、ヒューム管、L型擁壁、逆T式、RC、PC、BB\n"
    "6. 半角カタカナ（ｺﾝｸﾘｰﾄ等）は土木図書で慣用されるため、表記ゆれとして指摘しないこと\n"
    "7. 指摘がない場合は「指摘事項なし」とだけ回答すること\n"
    "\n"
    "出力形式（JSON配列）:\n"
    '[\n'
    '  {\n'
    '    "position": "該当箇所の前後5文字を含む引用（15文字程度）",\n'
    '    "type": "誤字|脱字|表記ゆれ|助詞|句読点",\n'
    '    "original": "原文の該当部分",\n'
    '    "suggested": "修正案",\n'
    '    "severity": "error|warning|info"\n'
    '  }\n'
    ']\n'
    "指摘事項がない場合: []"
)


@dataclass
class ProofreadingResult:
    """校正結果"""
    findings: List[Dict[str, Any]]
    raw_response: str
    backend: str  # "claude" or "ollama"
    success: bool
    error: Optional[str] = None


class ClaudeBackend:
    """Claude API を使用した校正バックエンド"""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-6"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic パッケージがインストールされていません。"
                    "pip install anthropic を実行してください。"
                )
        return self._client

    def is_available(self) -> bool:
        """Claude APIが利用可能か"""
        return bool(self.api_key)

    def proofread_chunk(self, chunk_text: str, chunk_index: int = 0,
                        total_chunks: int = 1) -> ProofreadingResult:
        """チャンクを校正する"""
        try:
            client = self._get_client()

            # 行番号を付与して位置特定精度を向上
            lines = chunk_text.split('\n')
            numbered_text = '\n'.join(
                f"L{i+1}: {line}" for i, line in enumerate(lines)
            )

            user_message = (
                f"以下の文章（チャンク {chunk_index + 1}/{total_chunks}）の"
                "誤字・脱字をチェックしてください。\n"
                "各行の先頭の「L数字:」は行番号です。"
                "positionフィールドには該当行の行番号（例: \"L15\"）を記入してください。\n"
                "---\n"
                f"{numbered_text}\n"
                "---"
            )

            response = client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=0.1,
                system=PROOFREADING_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}]
            )

            raw = response.content[0].text
            findings = self._parse_response(raw)

            return ProofreadingResult(
                findings=findings,
                raw_response=raw,
                backend="claude",
                success=True
            )

        except Exception as e:
            logger.error(f"Claude API エラー (chunk {chunk_index}): {e}")
            return ProofreadingResult(
                findings=[],
                raw_response="",
                backend="claude",
                success=False,
                error=str(e)
            )

    async def proofread_chunk_async(self, chunk_text: str, chunk_index: int = 0,
                                     total_chunks: int = 1) -> ProofreadingResult:
        """非同期版チャンク校正"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.proofread_chunk, chunk_text, chunk_index, total_chunks
        )

    def _parse_response(self, raw: str) -> List[Dict[str, Any]]:
        """Claude応答からJSON配列を抽出する"""
        # JSON配列部分を抽出
        raw_stripped = raw.strip()

        # ```json ... ``` ブロックの抽出
        import re
        json_block = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', raw_stripped, re.DOTALL)
        if json_block:
            raw_stripped = json_block.group(1)

        # 直接JSONとしてパース
        try:
            parsed = json.loads(raw_stripped)
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
        except json.JSONDecodeError:
            pass

        # JSON配列部分を探す
        start = raw_stripped.find('[')
        end = raw_stripped.rfind(']')
        if start >= 0 and end > start:
            try:
                parsed = json.loads(raw_stripped[start:end + 1])
                if isinstance(parsed, list):
                    return [item for item in parsed if isinstance(item, dict)]
            except json.JSONDecodeError:
                pass

        # パースできなかった場合：テキスト応答を1件の指摘として返す
        if raw_stripped and raw_stripped != "[]" and "指摘事項なし" not in raw_stripped:
            logger.warning("Claude応答をJSONとしてパースできません。テキストとして返します。")
            return [{"type": "info", "message": raw_stripped, "original": "", "suggested": ""}]

        return []


class OllamaChatBackend:
    """Ollama /api/chat を使用したフォールバックバックエンド

    従来の /api/generate (単一prompt) ではなく /api/chat (system/user ロール分離) を使い、
    校正指示の追従性を向上させる。
    """

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        self.base_url = base_url or self._detect_url()
        self.model = model or self._detect_best_model()

    def _detect_url(self) -> str:
        """接続可能なOllama URLを探す"""
        urls = [
            "http://host.docker.internal:11434",
            "http://localhost:11434",
            "http://172.17.0.1:11434",
        ]
        for url in urls:
            try:
                r = requests.get(f"{url}/api/tags", timeout=3)
                if r.status_code == 200:
                    return url
            except Exception:
                continue
        return "http://localhost:11434"

    def _detect_best_model(self) -> str:
        """利用可能な最大のモデルを選択（校正精度優先）"""
        preferred = ["qwen2.5:32b", "qwen2.5:14b", "llama3.2:3b"]
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if r.status_code == 200:
                available = [m["name"] for m in r.json().get("models", [])]
                for pref in preferred:
                    for avail in available:
                        if pref in avail:
                            logger.info(f"Ollama校正モデル選択: {avail}")
                            return avail
                if available:
                    return available[0]
        except Exception:
            pass
        return "llama3.2:3b"

    def is_available(self) -> bool:
        """Ollamaが利用可能か"""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def proofread_chunk(self, chunk_text: str, chunk_index: int = 0,
                        total_chunks: int = 1) -> ProofreadingResult:
        """/api/chat を使ってチャンクを校正する"""
        try:
            # 行番号を付与して位置特定精度を向上
            lines = chunk_text.split('\n')
            numbered_text = '\n'.join(
                f"L{i+1}: {line}" for i, line in enumerate(lines)
            )

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": PROOFREADING_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"以下の文章（チャンク {chunk_index + 1}/{total_chunks}）の"
                            "誤字・脱字をチェックしてください。\n"
                            "各行の先頭の「L数字:」は行番号です。"
                            "positionフィールドには該当行の行番号（例: \"L15\"）を記入してください。\n"
                            "---\n"
                            f"{numbered_text}\n"
                            "---"
                        )
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_ctx": 8192,
                    "num_predict": 4096,
                    "repeat_penalty": 1.3,
                }
            }

            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=300
            )

            if response.status_code == 200:
                data = response.json()
                raw = data.get("message", {}).get("content", "")
                # ClaudeBackendと同じパーサーを使用
                findings = ClaudeBackend._parse_response(None, raw)

                return ProofreadingResult(
                    findings=findings,
                    raw_response=raw,
                    backend="ollama",
                    success=True
                )
            else:
                return ProofreadingResult(
                    findings=[],
                    raw_response="",
                    backend="ollama",
                    success=False,
                    error=f"Ollama HTTP {response.status_code}"
                )

        except Exception as e:
            logger.error(f"Ollama校正エラー (chunk {chunk_index}): {e}")
            return ProofreadingResult(
                findings=[],
                raw_response="",
                backend="ollama",
                success=False,
                error=str(e)
            )

    async def proofread_chunk_async(self, chunk_text: str, chunk_index: int = 0,
                                     total_chunks: int = 1) -> ProofreadingResult:
        """非同期版チャンク校正"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.proofread_chunk, chunk_text, chunk_index, total_chunks
        )


def get_backend(preferred: str = "claude") -> Any:
    """設定に基づいてバックエンドを返す。Claude API優先、なければOllamaフォールバック"""
    if preferred == "claude":
        claude = ClaudeBackend()
        if claude.is_available():
            logger.info("Claude API バックエンドを使用")
            return claude
        logger.warning("ANTHROPIC_API_KEY未設定。Ollama /api/chat にフォールバック")

    ollama = OllamaChatBackend()
    if ollama.is_available():
        logger.info(f"Ollama /api/chat バックエンドを使用 (model={ollama.model})")
        return ollama

    raise RuntimeError("利用可能な校正バックエンドがありません。ANTHROPIC_API_KEY を設定するか、Ollama を起動してください。")
