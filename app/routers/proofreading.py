"""Proofreading endpoints (誤字・脱字校正 + 四則演算チェック).

GitHub側コミット 526d67f で main_unified.py に追加された校正機能を
ルーター構成へ移植したもの。
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import StreamingResponse

from ..dependencies import logger

router = APIRouter(prefix="/api", tags=["proofreading"])

# ---------------------------------------------------------------------------
# 校正専用モジュール
# ---------------------------------------------------------------------------
try:
    from src.proofreading import ProofreadingService, is_proofreading_request  # noqa: F401
    PROOFREADING_AVAILABLE = True
    logger.info("校正モジュールを読み込みました")
except ImportError as e:
    PROOFREADING_AVAILABLE = False
    logger.warning(f"校正モジュールが利用できません: {e}")

    # フォールバック: 簡易的な校正検出関数
    def is_proofreading_request(text: str) -> bool:
        if not text:
            return False
        keys = ["誤字", "脱字", "校正", "推敲", "表記ゆれ", "表記揺れ", "タイポ"]
        t = text.lower()
        return any(k in text for k in keys) or any(k in t for k in ["proofread", "proofreading", "typo"])


# ---------------------------------------------------------------------------
# 共有ヘルパー（生成系エンドポイントの校正モードでも使用）
# ---------------------------------------------------------------------------

def _truncate_chars(text: str, limit: int = 5200) -> str:
    """生成テキストを指定文字数に切り詰める"""
    if text and len(text) > limit:
        return text[:limit]
    return text


def _tokenizer_input_max_len(tokenizer, preferred: int) -> int:
    """tokenizerの上限を考慮した入力最大トークン長を返す"""
    model_limit = getattr(tokenizer, "model_max_length", 2048)
    if not isinstance(model_limit, int) or model_limit <= 0 or model_limit > 100000:
        model_limit = 4096
    return min(model_limit, preferred)


def _find_page_from_map(char_pos: int, page_map: list) -> int:
    """文字オフセットからページ番号を逆引きする"""
    for page_num, start, end in page_map:
        if start <= char_pos < end:
            return page_num
    return page_map[-1][0] if page_map else 1


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/proofread")
async def proofread_document_api(request: dict):
    """校正専用エンドポイント（誤字・脱字 + 四則演算・表の整合性チェック）"""
    if not PROOFREADING_AVAILABLE:
        return {"success": False, "error": "校正モジュールが利用できません"}

    text = request.get("text", "")
    if not text:
        return {"success": False, "error": "テキストが指定されていません"}

    check_arithmetic = request.get("check_arithmetic", True)
    backend_type = request.get("backend", "claude")

    try:
        service = ProofreadingService(
            check_arithmetic=check_arithmetic,
            backend_type=backend_type,
        )
        report = await service.proofread_document(text, check_arithmetic=check_arithmetic)
        return {"success": True, **report}
    except Exception as e:
        logger.error(f"校正エラー: {e}")
        return {"success": False, "error": str(e)}


@router.post("/proofread-pdf")
async def proofread_pdf_api(
    file: UploadFile = File(...),
    check_arithmetic: bool = Form(True),
    backend: str = Form("claude"),
):
    """PDF直接校正エンドポイント — ページ番号付きSSEストリーミング

    PDFファイルをサーバー側で直接テキスト抽出し校正する。
    プロンプト欄にテキストを挿入しないため、大量テキストでも操作性を維持できる。
    各指摘にはPDFのページ番号が付与される。
    """
    if not PROOFREADING_AVAILABLE:
        async def error_stream():
            yield f"data: {json.dumps({'event': 'error', 'message': '校正モジュールが利用できません'})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    if not file.filename.lower().endswith('.pdf'):
        async def error_stream():
            yield f"data: {json.dumps({'event': 'error', 'message': 'PDFファイルのみ対応しています'})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    # 1. PDF一時保存
    tmp_dir = Path("./temp_uploads/proofread")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{uuid.uuid4()}_{file.filename}"
    try:
        content = await file.read()
        with open(tmp_path, 'wb') as f:
            f.write(content)
    except Exception as e:
        async def error_stream():
            yield f"data: {json.dumps({'event': 'error', 'message': f'ファイル保存エラー: {e}'})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    # 2. ページ単位テキスト抽出 + page_map 構築
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(tmp_path))
        full_text = ""
        page_map = []  # [(page_num_1based, start_char, end_char), ...]
        for page_idx in range(len(doc)):
            # span座標ベースの行結合で数式を1行に保持する
            page = doc[page_idx]
            page_dict = page.get_text("dict")
            lines_by_y = {}  # y座標(生値) → [(x座標, テキスト), ...]
            for block in page_dict.get("blocks", []):
                for line in block.get("lines", []):
                    if not line.get("spans"):
                        continue
                    y_raw = round(line["bbox"][1], 1)
                    if y_raw not in lines_by_y:
                        lines_by_y[y_raw] = []
                    for span in line["spans"]:
                        span_text = span["text"].strip()
                        if span_text:
                            lines_by_y[y_raw].append((span["bbox"][0], span_text))
            # 近接y座標を統合（±3pt以内は同一行とみなす）
            merged_lines = {}
            for y in sorted(lines_by_y.keys()):
                merged_to = None
                for my in sorted(merged_lines.keys()):
                    if abs(y - my) <= 3.0:
                        merged_to = my
                        break
                if merged_to is not None:
                    merged_lines[merged_to].extend(lines_by_y[y])
                else:
                    merged_lines[y] = list(lines_by_y[y])
            # y座標順にソートし、各行のspanをx座標順に結合
            page_text_lines = []
            for y_key in sorted(merged_lines.keys()):
                spans = sorted(merged_lines[y_key], key=lambda s: s[0])
                line_text = " ".join(t for _, t in spans)
                page_text_lines.append(line_text)
            page_text = "\n".join(page_text_lines)
            if page_text.strip():
                start = len(full_text)
                full_text += page_text + "\n"
                end = len(full_text)
                page_map.append((page_idx + 1, start, end))
        doc.close()
        logger.info(f"PDF校正: {file.filename} → {len(page_map)}ページ, {len(full_text)}文字")
    except Exception as e:
        async def error_stream():
            yield f"data: {json.dumps({'event': 'error', 'message': f'PDF解析エラー: {e}'})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass

    if not full_text.strip():
        async def error_stream():
            yield f"data: {json.dumps({'event': 'error', 'message': 'PDFからテキストを抽出できませんでした。画像のみのPDFの可能性があります。'})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    # 3. 校正実行（SSEストリーミング）— findingsにページ番号を付与
    async def generate():
        try:
            service = ProofreadingService(
                check_arithmetic=check_arithmetic,
                backend_type=backend,
            )
            async for event in service.proofread_document_stream(full_text, check_arithmetic=check_arithmetic):
                # finding / arithmetic イベントにページ番号を付与
                if event.get("event") in ("finding", "arithmetic"):
                    data = event.get("data")
                    if isinstance(data, dict):
                        pos = data.get("position", 0)
                        data["page"] = _find_page_from_map(pos, page_map)

                # complete イベントの report 内の全 findings にもページ番号を付与
                if event.get("event") == "complete":
                    report = event.get("report", {})
                    for f_item in report.get("findings", []):
                        if isinstance(f_item, dict) and "page" not in f_item:
                            f_item["page"] = _find_page_from_map(f_item.get("position", 0), page_map)
                    report["total_pages"] = len(page_map)
                    report["filename"] = file.filename

                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"PDF校正ストリーミングエラー: {e}")
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/proofread-stream")
async def proofread_document_stream_api(request: dict):
    """校正ストリーミングエンドポイント（進捗をSSEでリアルタイム送信）"""
    if not PROOFREADING_AVAILABLE:
        async def error_stream():
            yield f"data: {json.dumps({'event': 'error', 'message': '校正モジュールが利用できません'})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    text = request.get("text", "")
    check_arithmetic = request.get("check_arithmetic", True)
    backend_type = request.get("backend", "claude")

    async def generate():
        try:
            service = ProofreadingService(
                check_arithmetic=check_arithmetic,
                backend_type=backend_type,
            )
            async for event in service.proofread_document_stream(text, check_arithmetic=check_arithmetic):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"校正ストリーミングエラー: {e}")
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
