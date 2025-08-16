"""
設定バリデーター
設定ファイルの整合性チェックと自動修正
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from .rag_config import RAGConfig
from .model_path_resolver import ModelPathResolver

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """検証問題"""
    level: str  # error, warning, info
    component: str  # llm, embedding, vector_store, etc.
    message: str
    suggested_fix: Optional[str] = None
    auto_fixable: bool = False


class ConfigValidator:
    """設定バリデーター"""
    
    def __init__(self, config: RAGConfig):
        """
        Args:
            config: 検証対象の設定
        """
        self.config = config
        self.issues: List[ValidationIssue] = []
        
    def validate_all(self) -> List[ValidationIssue]:
        """全設定を検証"""
        
        self.issues.clear()
        
        # 各コンポーネントを検証
        self._validate_llm_config()
        self._validate_embedding_config()
        self._validate_vector_store_config()
        self._validate_document_processing_config()
        self._validate_retrieval_config()
        
        return self.issues
        
    def _validate_llm_config(self):
        """LLM設定の検証"""
        
        llm_config = self.config.llm
        
        # ファインチューニング済みモデルのパス検証
        if llm_config.use_finetuned:
            path_issue = self._validate_model_path(
                llm_config.finetuned_model_path,
                "finetuned_model_path"
            )
            if path_issue:
                self.issues.append(path_issue)
                
        # メモリ設定の検証
        if llm_config.max_memory:
            total_memory = sum(
                int(mem.replace('GB', '')) 
                for mem in llm_config.max_memory.values() 
                if isinstance(mem, str) and 'GB' in mem
            )
            
            if total_memory > 128:  # 128GB以上は警告
                self.issues.append(ValidationIssue(
                    level="warning",
                    component="llm",
                    message=f"Large memory configuration: {total_memory}GB",
                    suggested_fix="Consider reducing max_memory if you don't have enough VRAM"
                ))
                
        # 生成パラメータの検証
        if not 0.0 <= llm_config.temperature <= 2.0:
            self.issues.append(ValidationIssue(
                level="error",
                component="llm", 
                message=f"Invalid temperature: {llm_config.temperature}",
                suggested_fix="Set temperature between 0.0 and 2.0",
                auto_fixable=True
            ))
            
    def _validate_model_path(self, model_path: str, config_key: str) -> Optional[ValidationIssue]:
        """モデルパスの検証"""
        
        resolver = ModelPathResolver()
        validation = resolver.validate_model_path(model_path)
        
        if not validation['is_valid']:
            # 代替モデルを探す
            alternative = resolver.find_latest_model()
            
            if alternative:
                return ValidationIssue(
                    level="warning",
                    component="llm",
                    message=f"Invalid model path: {model_path}. Issues: {', '.join(validation['issues'])}",
                    suggested_fix=f"Use detected model: {alternative}",
                    auto_fixable=True
                )
            else:
                return ValidationIssue(
                    level="error",
                    component="llm",
                    message=f"Invalid model path: {model_path}. No alternative models found.",
                    suggested_fix="Train a model first or set use_finetuned to false"
                )
                
        return None
        
    def _validate_embedding_config(self):
        """埋め込み設定の検証"""
        
        embedding_config = self.config.embedding
        
        # バッチサイズの検証
        if embedding_config.batch_size <= 0:
            self.issues.append(ValidationIssue(
                level="error",
                component="embedding",
                message=f"Invalid batch_size: {embedding_config.batch_size}",
                suggested_fix="Set batch_size to a positive integer",
                auto_fixable=True
            ))
            
        # 最大長の検証
        if not 128 <= embedding_config.max_length <= 2048:
            self.issues.append(ValidationIssue(
                level="warning",
                component="embedding",
                message=f"Unusual max_length: {embedding_config.max_length}",
                suggested_fix="Consider using 512 for most cases"
            ))
            
    def _validate_vector_store_config(self):
        """ベクトルストア設定の検証"""
        
        vs_config = self.config.vector_store
        
        # パスの検証
        vs_path = Path(vs_config.path)
        parent_dir = vs_path.parent
        
        if not parent_dir.exists():
            self.issues.append(ValidationIssue(
                level="warning",
                component="vector_store",
                message=f"Parent directory does not exist: {parent_dir}",
                suggested_fix="Parent directory will be created automatically"
            ))
            
    def _validate_document_processing_config(self):
        """文書処理設定の検証"""
        
        doc_config = self.config.document_processing
        
        # DPI設定の検証
        if not 150 <= doc_config.dpi <= 600:
            self.issues.append(ValidationIssue(
                level="warning",
                component="document_processing",
                message=f"Unusual DPI setting: {doc_config.dpi}",
                suggested_fix="Consider using 300 DPI for good quality"
            ))
            
    def _validate_retrieval_config(self):
        """検索設定の検証"""
        
        retrieval_config = self.config.retrieval
        
        # 重みの検証
        total_weight = retrieval_config.vector_weight + retrieval_config.keyword_weight
        
        if abs(total_weight - 1.0) > 0.01:  # 1.0から0.01以上離れている
            self.issues.append(ValidationIssue(
                level="warning",
                component="retrieval",
                message=f"Vector and keyword weights don't sum to 1.0: {total_weight}",
                suggested_fix="Adjust weights to sum to 1.0",
                auto_fixable=True
            ))
            
        # top_k設定の検証
        if retrieval_config.top_k <= 0:
            self.issues.append(ValidationIssue(
                level="error",
                component="retrieval",
                message=f"Invalid top_k: {retrieval_config.top_k}",
                suggested_fix="Set top_k to a positive integer",
                auto_fixable=True
            ))
            
    def auto_fix_issues(self) -> int:
        """自動修正可能な問題を修正"""
        
        fixed_count = 0
        
        for issue in self.issues:
            if not issue.auto_fixable:
                continue
                
            try:
                if issue.component == "llm":
                    if "temperature" in issue.message:
                        # 温度の修正
                        self.config.llm.temperature = max(0.0, min(2.0, self.config.llm.temperature))
                        fixed_count += 1
                        logger.info(f"Auto-fixed: {issue.message}")
                        
                    elif "model path" in issue.message.lower():
                        # モデルパスの修正
                        resolver = ModelPathResolver()
                        alternative = resolver.find_latest_model()
                        if alternative:
                            self.config.llm.finetuned_model_path = alternative
                            fixed_count += 1
                            logger.info(f"Auto-fixed model path: {alternative}")
                            
                elif issue.component == "embedding":
                    if "batch_size" in issue.message:
                        # バッチサイズの修正
                        self.config.embedding.batch_size = max(1, self.config.embedding.batch_size)
                        fixed_count += 1
                        
                elif issue.component == "retrieval":
                    if "weights" in issue.message:
                        # 重みの正規化
                        total = self.config.retrieval.vector_weight + self.config.retrieval.keyword_weight
                        if total > 0:
                            self.config.retrieval.vector_weight /= total
                            self.config.retrieval.keyword_weight /= total
                            fixed_count += 1
                            
                    elif "top_k" in issue.message:
                        # top_kの修正
                        self.config.retrieval.top_k = max(1, self.config.retrieval.top_k)
                        fixed_count += 1
                        
            except Exception as e:
                logger.warning(f"Failed to auto-fix issue: {issue.message} - {e}")
                
        return fixed_count
        
    def generate_report(self) -> str:
        """検証レポートを生成"""
        
        if not self.issues:
            return "✅ 設定に問題はありません"
            
        report_lines = ["🔍 設定検証レポート", "=" * 50]
        
        # レベル別に集計
        errors = [i for i in self.issues if i.level == "error"]
        warnings = [i for i in self.issues if i.level == "warning"]
        infos = [i for i in self.issues if i.level == "info"]
        
        # サマリー
        report_lines.append(f"エラー: {len(errors)}件")
        report_lines.append(f"警告: {len(warnings)}件") 
        report_lines.append(f"情報: {len(infos)}件")
        report_lines.append("")
        
        # 詳細
        for level_name, issues in [("エラー", errors), ("警告", warnings), ("情報", infos)]:
            if not issues:
                continue
                
            report_lines.append(f"【{level_name}】")
            for issue in issues:
                icon = "❌" if issue.level == "error" else "⚠️" if issue.level == "warning" else "ℹ️"
                report_lines.append(f"{icon} [{issue.component}] {issue.message}")
                
                if issue.suggested_fix:
                    fix_icon = "🔧" if issue.auto_fixable else "💡"
                    report_lines.append(f"   {fix_icon} {issue.suggested_fix}")
                    
                report_lines.append("")
                
        return "\n".join(report_lines)


def validate_config(config: RAGConfig, auto_fix: bool = True) -> Tuple[List[ValidationIssue], int]:
    """設定を検証（便利関数）
    
    Args:
        config: 検証対象の設定
        auto_fix: 自動修正を実行するか
        
    Returns:
        (検証問題のリスト, 修正された問題の数)
    """
    
    validator = ConfigValidator(config)
    issues = validator.validate_all()
    
    fixed_count = 0
    if auto_fix:
        fixed_count = validator.auto_fix_issues()
        
    return issues, fixed_count


def print_validation_report(config: RAGConfig, auto_fix: bool = True):
    """検証レポートを出力（便利関数）"""
    
    validator = ConfigValidator(config)
    issues = validator.validate_all()
    
    if auto_fix:
        fixed_count = validator.auto_fix_issues()
        if fixed_count > 0:
            print(f"🔧 {fixed_count}件の問題を自動修正しました")
            
        # 修正後に再検証
        issues = validator.validate_all()
        
    print(validator.generate_report())