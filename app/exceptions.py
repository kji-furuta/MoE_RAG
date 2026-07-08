"""アプリケーション固有の例外定義.

広範な ``except Exception`` を具体的な型に置き換えるための共通例外階層。
"""

from __future__ import annotations


class AppError(Exception):
    """アプリケーションエラーの基底クラス."""


# --- Model 関連 -----------------------------------------------------------

class ModelError(AppError):
    """モデル操作全般のエラー."""


class ModelLoadError(ModelError):
    """モデル読み込み失敗."""


class ModelNotFoundError(ModelError):
    """指定されたモデルが見つからない."""


class ModelMemoryError(ModelError):
    """GPU/CPUメモリ不足."""


# --- Training 関連 ---------------------------------------------------------

class TrainingError(AppError):
    """トレーニング処理のエラー."""


class TrainingConfigError(TrainingError):
    """トレーニング設定の不備."""


# --- RAG 関連 --------------------------------------------------------------

class RAGError(AppError):
    """RAGシステムのエラー."""


class RAGQueryError(RAGError):
    """RAGクエリ実行のエラー."""


class RAGIndexError(RAGError):
    """RAGインデックス操作のエラー."""


class RAGInitializationError(RAGError):
    """RAGシステム初期化のエラー."""


class DocumentProcessingError(RAGError):
    """文書処理のエラー."""


# --- Inference 関連 --------------------------------------------------------

class InferenceError(AppError):
    """推論処理のエラー."""


class GenerationError(InferenceError):
    """テキスト生成のエラー."""


class OllamaError(InferenceError):
    """Ollama連携のエラー."""


# --- Vector Store 関連 -----------------------------------------------------

class VectorStoreError(AppError):
    """ベクトルストア操作のエラー."""


class VectorStoreConnectionError(VectorStoreError):
    """ベクトルストア接続のエラー."""


# --- Upload / IO 関連 ------------------------------------------------------

class FileUploadError(AppError):
    """ファイルアップロードのエラー."""


class FileValidationError(FileUploadError):
    """ファイル検証のエラー."""
