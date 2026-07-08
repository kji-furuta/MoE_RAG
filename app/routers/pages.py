"""HTML page routes (template rendering)."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from starlette.requests import Request

from ..dependencies import templates
import app.dependencies as _deps

router = APIRouter(tags=["pages"])


@router.get("/")
async def root(request: Request):
    """メインページ"""
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/finetune")
async def finetune_page(request: Request):
    """ファインチューニング画面"""
    return templates.TemplateResponse("finetune.html", {"request": request})


@router.get("/models")
async def models_page(request: Request):
    """モデル一覧画面"""
    return templates.TemplateResponse("models.html", {"request": request})


@router.get("/readme")
async def readme_page(request: Request):
    """README.md表示ページ"""
    import markdown

    # ルートのREADME.mdを読み込む
    readme_path = Path(__file__).parent.parent.parent / "README.md"

    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_content = f.read()
        # MarkdownをHTMLに変換
        readme_html = markdown.markdown(
            readme_content,
            extensions=["extra", "codehilite", "tables", "toc"],
        )
    else:
        readme_html = "<p>README.mdファイルが見つかりません。</p>"

    return templates.TemplateResponse(
        "readme.html", {"request": request, "readme_content": readme_html}
    )


@router.get("/system-overview", response_class=HTMLResponse)
async def system_overview_page(request: Request):
    """システム全体の概要ページ"""
    context = {
        "request": request,
        "rag_available": _deps.RAG_AVAILABLE,
    }
    return templates.TemplateResponse("system-overview.html", context)


@router.get("/rag")
async def rag_page(request: Request):
    """RAGシステム画面"""
    return templates.TemplateResponse(
        "rag.html", {"request": request, "rag_available": _deps.RAG_AVAILABLE}
    )


@router.get("/rlanything")
async def rlanything_page(request: Request):
    """RLAnything GRPO強化学習画面"""
    return templates.TemplateResponse("rlanything.html", {"request": request})


@router.get("/dpo")
async def dpo_page(request: Request):
    """DPO Preference収集画面"""
    return templates.TemplateResponse("dpo.html", {"request": request})
