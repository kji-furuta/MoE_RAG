"""
埋め込みモデル管理モジュール
日本語対応の高性能な埋め込みモデルを提供
"""

import os
from typing import List, Union, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from loguru import logger
import hashlib
import json
from pathlib import Path


class EmbeddingModel:
    """埋め込みモデルの基底クラス"""
    
    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._cache_dir = Path("./cache/embeddings")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        
    def encode(self, texts: Union[str, List[str]], 
              batch_size: int = 32, 
              show_progress: bool = False) -> np.ndarray:
        """テキストを埋め込みベクトルに変換"""
        raise NotImplementedError
        
    def _get_cache_key(self, text: str) -> str:
        """テキストのキャッシュキーを生成"""
        return hashlib.md5(f"{self.model_name}:{text}".encode()).hexdigest()
        
    def _load_from_cache(self, texts: List[str]) -> dict:
        """キャッシュから埋め込みを読み込み"""
        cached = {}
        for text in texts:
            cache_key = self._get_cache_key(text)
            cache_file = self._cache_dir / f"{cache_key}.npy"
            if cache_file.exists():
                cached[text] = np.load(cache_file)
        return cached
        
    def _save_to_cache(self, text: str, embedding: np.ndarray):
        """埋め込みをキャッシュに保存"""
        cache_key = self._get_cache_key(text)
        cache_file = self._cache_dir / f"{cache_key}.npy"
        np.save(cache_file, embedding)


class MultilingualE5EmbeddingModel(EmbeddingModel):
    """intfloat/multilingual-e5-large 埋め込みモデル"""
    
    def __init__(self, 
                 model_name: str = "intfloat/multilingual-e5-large",
                 device: Optional[str] = None,
                 max_length: int = 512,
                 normalize_embeddings: bool = True):
        super().__init__(model_name, device)
        
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings
        
        # モデルとトークナイザーの読み込み
        logger.info(f"Loading embedding model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # E5モデルは特別なプレフィックスが必要
        self.query_prefix = "query: "
        self.passage_prefix = "passage: "
        
    def encode(self, 
              texts: Union[str, List[str]], 
              batch_size: int = 32,
              show_progress: bool = False,
              is_query: bool = False) -> np.ndarray:
        """
        テキストを埋め込みベクトルに変換
        
        Args:
            texts: 変換するテキスト（文字列またはリスト）
            batch_size: バッチサイズ
            show_progress: 進捗表示
            is_query: クエリテキストかどうか（E5モデル用）
        """
        if isinstance(texts, str):
            texts = [texts]
            
        # プレフィックスを追加
        prefix = self.query_prefix if is_query else self.passage_prefix
        texts = [prefix + text for text in texts]
        
        # キャッシュから読み込み
        cached_embeddings = self._load_from_cache(texts)
        texts_to_encode = [t for t in texts if t not in cached_embeddings]
        
        embeddings_list = []
        
        if texts_to_encode:
            # バッチ処理
            for i in range(0, len(texts_to_encode), batch_size):
                batch_texts = texts_to_encode[i:i + batch_size]
                
                # トークナイズ
                encoded_input = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                # 埋め込み生成
                with torch.no_grad():
                    model_output = self.model(**encoded_input)
                    # Mean pooling
                    embeddings = self._mean_pooling(
                        model_output, 
                        encoded_input['attention_mask']
                    )
                    
                    if self.normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                        
                embeddings_np = embeddings.cpu().numpy()
                
                # キャッシュに保存
                for text, embedding in zip(batch_texts, embeddings_np):
                    self._save_to_cache(text, embedding)
                    
                embeddings_list.extend(embeddings_np)
                
        # 結果を統合
        all_embeddings = []
        for text in texts:
            if text in cached_embeddings:
                all_embeddings.append(cached_embeddings[text])
            else:
                # texts_to_encodeのインデックスを見つけて対応する埋め込みを取得
                idx = texts_to_encode.index(text)
                all_embeddings.append(embeddings_list[idx])
                
        return np.array(all_embeddings)
        
    def _mean_pooling(self, model_output, attention_mask):
        """Mean poolingを実行"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )


class SentenceTransformerEmbeddingModel(EmbeddingModel):
    """SentenceTransformersベースの埋め込みモデル"""
    
    def __init__(self,
                 model_name: str = "sonoisa/sentence-bert-base-ja-mean-tokens-v2",
                 device: Optional[str] = None,
                 normalize_embeddings: bool = True):
        super().__init__(model_name, device)
        
        self.normalize_embeddings = normalize_embeddings
        
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device)
        
    def encode(self,
              texts: Union[str, List[str]],
              batch_size: int = 32,
              show_progress: bool = False) -> np.ndarray:
        """テキストを埋め込みベクトルに変換"""
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True
        )
        
        return embeddings


class EmbeddingModelFactory:
    """埋め込みモデルのファクトリクラス"""
    
    # 利用可能なモデル
    AVAILABLE_MODELS = {
        "multilingual-e5-large": {
            "class": MultilingualE5EmbeddingModel,
            "model_name": "intfloat/multilingual-e5-large",
            "embedding_dim": 1024
        },
        "multilingual-e5-base": {
            "class": MultilingualE5EmbeddingModel,
            "model_name": "intfloat/multilingual-e5-base",
            "embedding_dim": 768
        },
        "multilingual-e5-small": {
            "class": MultilingualE5EmbeddingModel,
            "model_name": "intfloat/multilingual-e5-small",
            "embedding_dim": 384
        },
        "sentence-bert-ja": {
            "class": SentenceTransformerEmbeddingModel,
            "model_name": "sonoisa/sentence-bert-base-ja-mean-tokens-v2",
            "embedding_dim": 768
        },
        "sup-simcse-ja": {
            "class": SentenceTransformerEmbeddingModel,
            "model_name": "cl-nagoya/sup-simcse-ja-large",
            "embedding_dim": 1024
        }
    }
    
    @classmethod
    def create(cls, 
              model_type: str = "multilingual-e5-large",
              device: Optional[str] = None,
              **kwargs) -> EmbeddingModel:
        """埋め込みモデルのインスタンスを作成"""
        
        if model_type not in cls.AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available models: {list(cls.AVAILABLE_MODELS.keys())}"
            )
            
        model_config = cls.AVAILABLE_MODELS[model_type]
        model_class = model_config["class"]
        model_name = model_config["model_name"]
        
        return model_class(model_name=model_name, device=device, **kwargs)
        
    @classmethod
    def get_embedding_dim(cls, model_type: str) -> int:
        """モデルの埋め込み次元数を取得"""
        if model_type not in cls.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model type: {model_type}")
            
        return cls.AVAILABLE_MODELS[model_type]["embedding_dim"]


# 便利な関数
def get_embedding_model(model_type: str = "multilingual-e5-large", 
                       **kwargs) -> EmbeddingModel:
    """埋め込みモデルを取得"""
    return EmbeddingModelFactory.create(model_type, **kwargs)


def batch_encode_texts(texts: List[str],
                      model: Optional[EmbeddingModel] = None,
                      batch_size: int = 32,
                      show_progress: bool = True) -> np.ndarray:
    """テキストのリストをバッチで埋め込みに変換"""
    if model is None:
        model = get_embedding_model()
        
    return model.encode(texts, batch_size=batch_size, show_progress=show_progress)