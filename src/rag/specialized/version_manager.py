"""
バージョン管理システム
文書のバージョン管理と差分検出機能
"""

import hashlib
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import difflib
import logging
logger = logging.getLogger(__name__)
import sqlite3
from contextlib import contextmanager


@dataclass
class DocumentVersion:
    """文書バージョン"""
    document_id: str
    version_id: str
    version_number: str
    title: str
    content_hash: str
    created_at: datetime
    updated_at: datetime
    file_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    parent_version_id: Optional[str] = None
    change_summary: Optional[str] = None
    status: str = "active"  # active, archived, deprecated
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            'document_id': self.document_id,
            'version_id': self.version_id,
            'version_number': self.version_number,
            'title': self.title,
            'content_hash': self.content_hash,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'file_path': self.file_path,
            'metadata': self.metadata,
            'parent_version_id': self.parent_version_id,
            'change_summary': self.change_summary,
            'status': self.status
        }


@dataclass
class VersionDiff:
    """バージョン間の差分"""
    old_version_id: str
    new_version_id: str
    diff_type: str  # content, metadata, structure
    changes: List[Dict[str, Any]]
    added_sections: List[str]
    removed_sections: List[str]
    modified_sections: List[str]
    similarity_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return asdict(self)


class VersionDatabase:
    """バージョン管理データベース"""
    
    def __init__(self, db_path: str = "./version_db.sqlite"):
        """
        Args:
            db_path: データベースファイルのパス
        """
        self.db_path = db_path
        self._init_database()
        
    def _init_database(self):
        """データベースを初期化"""
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # バージョンテーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    version_id TEXT UNIQUE NOT NULL,
                    version_number TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    file_path TEXT,
                    parent_version_id TEXT,
                    change_summary TEXT,
                    status TEXT DEFAULT 'active',
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (parent_version_id) REFERENCES document_versions(version_id)
                )
            """)
            
            # インデックス
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_id 
                ON document_versions(document_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_version_number 
                ON document_versions(version_number)
            """)
            
            # 変更履歴テーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS version_changes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    old_version_id TEXT NOT NULL,
                    new_version_id TEXT NOT NULL,
                    change_type TEXT NOT NULL,
                    change_details TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (old_version_id) REFERENCES document_versions(version_id),
                    FOREIGN KEY (new_version_id) REFERENCES document_versions(version_id)
                )
            """)
            
            conn.commit()
            
    @contextmanager
    def _get_connection(self):
        """データベース接続を取得"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
            
    def add_version(self, version: DocumentVersion) -> str:
        """バージョンを追加"""
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            metadata_json = json.dumps(version.metadata) if version.metadata else None
            
            cursor.execute("""
                INSERT INTO document_versions (
                    document_id, version_id, version_number, title,
                    content_hash, file_path, parent_version_id,
                    change_summary, status, metadata, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                version.document_id,
                version.version_id,
                version.version_number,
                version.title,
                version.content_hash,
                version.file_path,
                version.parent_version_id,
                version.change_summary,
                version.status,
                metadata_json,
                version.created_at,
                version.updated_at
            ))
            
            conn.commit()
            
        return version.version_id
        
    def get_version(self, version_id: str) -> Optional[DocumentVersion]:
        """バージョンを取得"""
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM document_versions WHERE version_id = ?
            """, (version_id,))
            
            row = cursor.fetchone()
            
            if row:
                return self._row_to_version(row)
                
        return None
        
    def get_document_versions(self, document_id: str) -> List[DocumentVersion]:
        """文書のすべてのバージョンを取得"""
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM document_versions 
                WHERE document_id = ?
                ORDER BY created_at DESC
            """, (document_id,))
            
            rows = cursor.fetchall()
            
        return [self._row_to_version(row) for row in rows]
        
    def get_latest_version(self, document_id: str) -> Optional[DocumentVersion]:
        """最新バージョンを取得"""
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM document_versions 
                WHERE document_id = ? AND status = 'active'
                ORDER BY created_at DESC
                LIMIT 1
            """, (document_id,))
            
            row = cursor.fetchone()
            
            if row:
                return self._row_to_version(row)
                
        return None
        
    def _row_to_version(self, row: sqlite3.Row) -> DocumentVersion:
        """データベース行をDocumentVersionに変換"""
        
        metadata = json.loads(row['metadata']) if row['metadata'] else None
        
        return DocumentVersion(
            document_id=row['document_id'],
            version_id=row['version_id'],
            version_number=row['version_number'],
            title=row['title'],
            content_hash=row['content_hash'],
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at']),
            file_path=row['file_path'],
            metadata=metadata,
            parent_version_id=row['parent_version_id'],
            change_summary=row['change_summary'],
            status=row['status']
        )


class DiffDetector:
    """差分検出エンジン"""
    
    def __init__(self):
        """差分検出エンジンの初期化"""
        
        # セクション識別パターン
        self.section_patterns = [
            r'^第[0-9０-９]+章',
            r'^第[0-9０-９]+節',
            r'^第[0-9０-９]+項',
            r'^[0-9]+\.',
            r'^[0-9]+\.[0-9]+',
            r'^（[0-9０-９]+）',
            r'^[一二三四五六七八九十]+、'
        ]
        
    def detect_changes(self, old_content: str, new_content: str) -> VersionDiff:
        """コンテンツ間の変更を検出"""
        
        # セクション単位で分割
        old_sections = self._split_into_sections(old_content)
        new_sections = self._split_into_sections(new_content)
        
        # セクションレベルの差分
        added_sections = []
        removed_sections = []
        modified_sections = []
        
        old_section_keys = set(old_sections.keys())
        new_section_keys = set(new_sections.keys())
        
        # 追加されたセクション
        for key in new_section_keys - old_section_keys:
            added_sections.append(key)
            
        # 削除されたセクション
        for key in old_section_keys - new_section_keys:
            removed_sections.append(key)
            
        # 変更されたセクション
        for key in old_section_keys & new_section_keys:
            if old_sections[key] != new_sections[key]:
                modified_sections.append(key)
                
        # 詳細な差分
        changes = self._get_detailed_changes(old_content, new_content)
        
        # 類似度スコア
        similarity_score = self._calculate_similarity(old_content, new_content)
        
        return VersionDiff(
            old_version_id="",  # 後で設定
            new_version_id="",  # 後で設定
            diff_type="content",
            changes=changes,
            added_sections=added_sections,
            removed_sections=removed_sections,
            modified_sections=modified_sections,
            similarity_score=similarity_score
        )
        
    def _split_into_sections(self, content: str) -> Dict[str, str]:
        """コンテンツをセクション単位で分割"""
        
        import re
        
        sections = {}
        current_section = "前文"
        current_content = []
        
        for line in content.split('\n'):
            # セクションヘッダーを検出
            is_section_header = False
            
            for pattern in self.section_patterns:
                if re.match(pattern, line.strip()):
                    # 前のセクションを保存
                    if current_content:
                        sections[current_section] = '\n'.join(current_content)
                        
                    current_section = line.strip()
                    current_content = [line]
                    is_section_header = True
                    break
                    
            if not is_section_header:
                current_content.append(line)
                
        # 最後のセクションを保存
        if current_content:
            sections[current_section] = '\n'.join(current_content)
            
        return sections
        
    def _get_detailed_changes(self, old_content: str, new_content: str) -> List[Dict[str, Any]]:
        """詳細な変更内容を取得"""
        
        old_lines = old_content.splitlines()
        new_lines = new_content.splitlines()
        
        differ = difflib.unified_diff(
            old_lines, new_lines,
            lineterm='', n=3
        )
        
        changes = []
        current_change = None
        
        for line in differ:
            if line.startswith('@@'):
                # 新しい変更ブロック
                if current_change:
                    changes.append(current_change)
                    
                # 行番号を解析
                match = re.match(r'@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@', line)
                if match:
                    current_change = {
                        'type': 'modification',
                        'old_start': int(match.group(1)),
                        'old_count': int(match.group(2) or 1),
                        'new_start': int(match.group(3)),
                        'new_count': int(match.group(4) or 1),
                        'removed_lines': [],
                        'added_lines': []
                    }
                    
            elif line.startswith('-') and not line.startswith('---'):
                if current_change:
                    current_change['removed_lines'].append(line[1:])
                    
            elif line.startswith('+') and not line.startswith('+++'):
                if current_change:
                    current_change['added_lines'].append(line[1:])
                    
        # 最後の変更を追加
        if current_change:
            changes.append(current_change)
            
        return changes
        
    def _calculate_similarity(self, old_content: str, new_content: str) -> float:
        """コンテンツの類似度を計算"""
        
        # 文字レベルの類似度
        matcher = difflib.SequenceMatcher(None, old_content, new_content)
        return matcher.ratio()


class ChangeTracker:
    """変更履歴トラッキング"""
    
    def __init__(self, db: VersionDatabase):
        """
        Args:
            db: バージョンデータベース
        """
        self.db = db
        self.detector = DiffDetector()
        
    def track_change(self, old_version_id: str, new_version_id: str,
                    old_content: str, new_content: str) -> VersionDiff:
        """バージョン間の変更を追跡"""
        
        # 差分を検出
        diff = self.detector.detect_changes(old_content, new_content)
        diff.old_version_id = old_version_id
        diff.new_version_id = new_version_id
        
        # データベースに記録
        self._record_change(diff)
        
        return diff
        
    def _record_change(self, diff: VersionDiff):
        """変更をデータベースに記録"""
        
        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            
            change_details = json.dumps(diff.to_dict())
            
            cursor.execute("""
                INSERT INTO version_changes (
                    old_version_id, new_version_id, change_type, change_details
                ) VALUES (?, ?, ?, ?)
            """, (
                diff.old_version_id,
                diff.new_version_id,
                diff.diff_type,
                change_details
            ))
            
            conn.commit()
            
    def get_change_history(self, document_id: str) -> List[Dict[str, Any]]:
        """文書の変更履歴を取得"""
        
        versions = self.db.get_document_versions(document_id)
        
        history = []
        for i in range(len(versions) - 1):
            new_version = versions[i]
            old_version = versions[i + 1]
            
            with self.db._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM version_changes
                    WHERE old_version_id = ? AND new_version_id = ?
                """, (old_version.version_id, new_version.version_id))
                
                row = cursor.fetchone()
                
                if row:
                    change_details = json.loads(row['change_details'])
                    history.append({
                        'old_version': old_version.to_dict(),
                        'new_version': new_version.to_dict(),
                        'changes': change_details,
                        'changed_at': row['created_at']
                    })
                    
        return history


class VersionManager:
    """統合バージョン管理システム"""
    
    def __init__(self, db_path: str = "./version_db.sqlite"):
        """
        Args:
            db_path: データベースファイルのパス
        """
        self.db = VersionDatabase(db_path)
        self.tracker = ChangeTracker(self.db)
        
    def create_version(self, document_id: str, title: str, content: str,
                      file_path: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None,
                      parent_version_id: Optional[str] = None) -> DocumentVersion:
        """新しいバージョンを作成"""
        
        # コンテンツハッシュを計算
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        # バージョン番号を生成
        version_number = self._generate_version_number(document_id)
        
        # バージョンIDを生成
        version_id = f"{document_id}_v{version_number}_{content_hash[:8]}"
        
        # バージョンオブジェクトを作成
        version = DocumentVersion(
            document_id=document_id,
            version_id=version_id,
            version_number=version_number,
            title=title,
            content_hash=content_hash,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            file_path=file_path,
            metadata=metadata,
            parent_version_id=parent_version_id,
            change_summary=None,
            status="active"
        )
        
        # データベースに追加
        self.db.add_version(version)
        
        # 親バージョンがある場合は差分を検出
        if parent_version_id:
            parent_version = self.db.get_version(parent_version_id)
            if parent_version and parent_version.file_path:
                # ファイルからコンテンツを読み込んで差分検出
                # （簡略化のため、実装は省略）
                pass
                
        logger.info(f"Created new version: {version_id}")
        return version
        
    def _generate_version_number(self, document_id: str) -> str:
        """バージョン番号を生成"""
        
        versions = self.db.get_document_versions(document_id)
        
        if not versions:
            return "1.0.0"
            
        # 最新バージョン番号を解析
        latest = versions[0]
        parts = latest.version_number.split('.')
        
        if len(parts) == 3:
            major, minor, patch = map(int, parts)
            # パッチバージョンをインクリメント
            return f"{major}.{minor}.{patch + 1}"
        else:
            return "1.0.0"
            
    def compare_versions(self, version_id1: str, version_id2: str,
                        content1: str, content2: str) -> VersionDiff:
        """2つのバージョンを比較"""
        
        return self.tracker.track_change(version_id1, version_id2, content1, content2)
        
    def get_version_tree(self, document_id: str) -> Dict[str, Any]:
        """バージョンツリーを取得"""
        
        versions = self.db.get_document_versions(document_id)
        
        # バージョンをツリー構造に変換
        version_dict = {v.version_id: v for v in versions}
        tree = {'root': [], 'nodes': {}}
        
        for version in versions:
            if version.parent_version_id is None:
                tree['root'].append(version.version_id)
            else:
                parent_id = version.parent_version_id
                if parent_id not in tree['nodes']:
                    tree['nodes'][parent_id] = []
                tree['nodes'][parent_id].append(version.version_id)
                
        return {
            'document_id': document_id,
            'tree': tree,
            'versions': {v.version_id: v.to_dict() for v in versions}
        }
        
    def archive_old_versions(self, document_id: str, keep_latest: int = 5):
        """古いバージョンをアーカイブ"""
        
        versions = self.db.get_document_versions(document_id)
        
        # アクティブなバージョンのみ
        active_versions = [v for v in versions if v.status == 'active']
        
        if len(active_versions) > keep_latest:
            # 古いバージョンをアーカイブ
            to_archive = active_versions[keep_latest:]
            
            with self.db._get_connection() as conn:
                cursor = conn.cursor()
                
                for version in to_archive:
                    cursor.execute("""
                        UPDATE document_versions
                        SET status = 'archived', updated_at = ?
                        WHERE version_id = ?
                    """, (datetime.now(), version.version_id))
                    
                conn.commit()
                
            logger.info(f"Archived {len(to_archive)} old versions of {document_id}")


# 便利な関数
def create_version_manager(db_path: str = "./version_db.sqlite") -> VersionManager:
    """バージョン管理システムを作成"""
    return VersionManager(db_path)


def detect_changes(old_content: str, new_content: str) -> VersionDiff:
    """コンテンツ間の変更を検出（便利関数）"""
    detector = DiffDetector()
    return detector.detect_changes(old_content, new_content)