# AcceleratorState競合エラーの分析レポート

**日付**: 2025-10-02
**問題**: ファインチューニング後に継続学習を実行すると `AcceleratorState has already been initialized` エラー発生
**調査者**: Claude Code + Codex MCP (進行中)

---

## 問題の症状

### 正常動作パターン
```
システム再起動 → 継続学習実行 → ✅ 成功
```

### エラー発生パターン
```
ファインチューニング実行 → 継続学習実行 → ❌ AcceleratorStateエラー
```

**エラーメッセージ**:
```python
ValueError: AcceleratorState has already been initialized and cannot be changed,
restart your runtime completely and pass `mixed_precision='bf16'` to `Accelerator()`.
```

**発生箇所**: `src/training/ewc_full_finetuning.py:80`
```python
self.accelerator = Accelerator(
    gradient_accumulation_steps=self.config.gradient_accumulation_steps,
    mixed_precision=self.mixed_precision_mode,  # ← ここでエラー
    log_with="wandb" if os.environ.get("WANDB_API_KEY") else None,
)
```

---

## 根本原因の特定

### AcceleratorStateの仕組み

Hugging Face Accelerateライブラリの`AcceleratorState`は**シングルトンパターン**で実装されています。

```python
# accelerate/state.py (簡略化)
class AcceleratorState:
    _shared_state = {}  # クラス変数（全インスタンスで共有）

    def __init__(self, mixed_precision=None, ...):
        # 既に初期化されている場合、設定変更は不可
        if self._shared_state:
            if mixed_precision != self._shared_state['mixed_precision']:
                raise ValueError("AcceleratorState has already been initialized...")
```

### エラー発生のフロー

#### 1. ファインチューニング実行 (成功)
```
[app/training/service.py:run_training_task]
  ↓
[src/training/lora_finetuning.py:__init__]
  ↓
self.accelerator = Accelerator(mixed_precision="fp16")  # ← AcceleratorState._shared_state初期化
  ↓
訓練完了
  ↓
関数終了 ← ❌ AcceleratorStateがクリアされない！
```

**問題**: `run_training_task`関数終了時に`AcceleratorState._reset_state()`が呼ばれていない

#### 2. 継続学習実行 (エラー)
```
[app/continual_learning/continual_learning_ui.py:run_continual_learning_task]
  ↓
[src/training/continual_learning_pipeline.py:run_continual_task]
  ↓
[src/training/ewc_full_finetuning.py:__init__]
  ↓
self.accelerator = Accelerator(mixed_precision="bf16")  # ← ❌ エラー！
```

**理由**: `AcceleratorState._shared_state`に`mixed_precision="fp16"`が残っている
**競合**: 新しく`mixed_precision="bf16"`で初期化しようとして拒否される

#### 3. システム再起動後 (成功)
```
プロセス再起動 → AcceleratorState._shared_state = {} (クリア)
  ↓
継続学習実行 → ✅ 成功（初回初期化のため）
```

---

## コード箇所の詳細

### 問題箇所1: LoRAファインチューニング（クリーンアップ不足）

**ファイル**: `src/training/lora_finetuning.py:84`
```python
class LoRAFinetuningTrainer:
    def __init__(self, model, lora_config, training_config, ...):
        # Acceleratorの初期化
        self.accelerator = Accelerator(
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            mixed_precision="fp16" if training_config.fp16 else "no",  # ← fp16
            log_with="wandb" if os.environ.get("WANDB_API_KEY") else None,
        )

    # ❌ __del__やcleanup()メソッドが存在しない
    # ❌ AcceleratorState._reset_state()が呼ばれない
```

### 問題箇所2: 継続学習トレーナー（異なる設定で初期化）

**ファイル**: `src/training/ewc_full_finetuning.py:80`
```python
class EWCFullFinetuningTrainer:
    def __init__(self, model, config, ewc_lambda, ...):
        # Acceleratorの初期化
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision_mode,  # ← bf16（異なる！）
            log_with="wandb" if os.environ.get("WANDB_API_KEY") else None,
        )
```

**`_resolve_mixed_precision_mode()`の結果**: `"bf16"`を返す（32Bモデル用）

### 呼び出し元: ファインチューニングサービス

**ファイル**: `app/training/service.py:74-629`
```python
async def run_training_task(task_id: str, request: TrainingRequest):
    # ... 省略 ...

    # LoRAトレーナー作成
    trainer = LoRAFinetuningTrainer(
        model=model,
        lora_config=lora_config,
        training_config=training_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )  # ← AcceleratorState._shared_state['mixed_precision'] = 'fp16'

    # 訓練実行
    train_result = trainer.train()

    # モデル保存
    model.save_pretrained(str(output_dir))

    # ❌ AcceleratorState._reset_state()が呼ばれていない
    # 関数終了
```

---

## 解決策の提案

### 解決策1: ファインチューニング終了時にAcceleratorStateをリセット（推奨）

**修正箇所**: `app/training/service.py:run_training_task`関数の最後

```python
async def run_training_task(task_id: str, request: TrainingRequest):
    try:
        # ... 既存のトレーニング処理 ...

        # モデル保存
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        # 完了
        training_tasks[task_id].status = "completed"
        training_tasks[task_id].message = f"{method_name}ファインチューニング完了！"

    finally:
        # AcceleratorStateのクリーンアップ（重要！）
        from accelerate.state import AcceleratorState
        AcceleratorState._reset_state(reset_partial_state=True)
        logger.info(f"Task {task_id}: AcceleratorStateをリセットしました")
```

**利点**:
- ✅ 他のコードへの影響最小限
- ✅ 継続学習側の変更不要
- ✅ 将来的な他のトレーナー追加にも対応

**欠点**:
- ⚠️ `_reset_state()`は内部APIのため、将来のAccelerateバージョンで変更される可能性

### 解決策2: 既存AcceleratorStateの再利用

**修正箇所**: `src/training/ewc_full_finetuning.py:__init__`

```python
def __init__(self, model, config, ewc_lambda, ...):
    # ... 省略 ...

    # Acceleratorの初期化または再利用
    from accelerate import Accelerator
    from accelerate.state import AcceleratorState

    # 既存のAcceleratorStateがある場合はリセット
    if AcceleratorState._shared_state:
        logger.warning("AcceleratorStateが既に初期化されています。リセットします。")
        AcceleratorState._reset_state(reset_partial_state=True)

    self.accelerator = Accelerator(
        gradient_accumulation_steps=self.config.gradient_accumulation_steps,
        mixed_precision=self.mixed_precision_mode,
        log_with="wandb" if os.environ.get("WANDB_API_KEY") else None,
    )
```

**利点**:
- ✅ 防御的プログラミング（どの順序で呼ばれても動作）
- ✅ 継続学習側で問題を自己修復

**欠点**:
- ⚠️ 問題の根本原因を解決していない
- ⚠️ ファインチューニング側でもAcceleratorを使用中の場合、競合の可能性

### 解決策3: コンテキストマネージャーでAccelerator管理

**修正箇所**: 新しいユーティリティクラス作成

```python
# src/training/accelerator_manager.py (新規)
from contextlib import contextmanager
from accelerate import Accelerator
from accelerate.state import AcceleratorState

@contextmanager
def managed_accelerator(**kwargs):
    """
    AcceleratorStateを自動クリーンアップするコンテキストマネージャー

    with managed_accelerator(mixed_precision='fp16') as accelerator:
        # トレーニング処理
        pass
    # 自動的にAcceleratorState._reset_state()が呼ばれる
    """
    accelerator = None
    try:
        accelerator = Accelerator(**kwargs)
        yield accelerator
    finally:
        if accelerator is not None:
            AcceleratorState._reset_state(reset_partial_state=True)
```

**使用例**:
```python
# src/training/lora_finetuning.py
def __init__(self, ...):
    # Acceleratorは外部で管理されることを想定
    self.accelerator = None

def train_with_accelerator(self, accelerator):
    self.accelerator = accelerator
    # 訓練処理
    return self.train()

# app/training/service.py
with managed_accelerator(mixed_precision='fp16') as accelerator:
    trainer = LoRAFinetuningTrainer(...)
    result = trainer.train_with_accelerator(accelerator)
# AcceleratorStateは自動クリーンアップされる
```

**利点**:
- ✅ Pythonic（withステートメント）
- ✅ リソース管理が明示的
- ✅ 例外発生時も確実にクリーンアップ

**欠点**:
- ⚠️ 既存コードの大幅な変更が必要

---

## 推奨する修正手順

### フェーズ1: 緊急対応（解決策1）

1. **`app/training/service.py`にfinally句を追加**
   - `run_training_task`関数の最後に`AcceleratorState._reset_state()`を呼び出し
   - 所要時間: 5分
   - リスク: 低

2. **単体テスト作成**
   - ファインチューニング→継続学習の順序実行テスト
   - AcceleratorStateがクリアされることを確認
   - 所要時間: 15分

3. **統合テスト**
   - 実際のDocker環境でエンドツーエンドテスト
   - 所要時間: 10分

### フェーズ2: 防御的実装（解決策2）

1. **`src/training/ewc_full_finetuning.py`に防御コード追加**
   - 既存AcceleratorStateの検出とリセット
   - 所要時間: 10分
   - リスク: 低

2. **同様の修正を他のトレーナーにも適用**
   - `src/training/full_finetuning.py`
   - `src/training/multi_gpu_training.py`
   - 所要時間: 15分

### フェーズ3: 長期的改善（解決策3）

1. **`managed_accelerator`コンテキストマネージャー実装**
   - 新規ファイル作成
   - 所要時間: 20分

2. **既存コードをリファクタリング**
   - 段階的に移行（破壊的変更なし）
   - 所要時間: 1-2時間

---

## 影響範囲の評価

### 直接影響
- ✅ LoRAファインチューニング → 継続学習の順序実行
- ✅ 継続学習の繰り返し実行

### 間接影響（要確認）
- ⚠️ マルチGPU訓練との競合
- ⚠️ DeepSpeed統合との競合
- ⚠️ Wandbロギングへの影響

### 影響なし
- ✅ 単独のファインチューニング実行
- ✅ 単独の継続学習実行（システム再起動後）
- ✅ RAGシステム
- ✅ モデル推論

---

## テスト計画

### 単体テスト
```python
# tests/test_accelerator_state_cleanup.py
def test_finetuning_clears_accelerator_state():
    """ファインチューニング後にAcceleratorStateがクリアされることを検証"""
    from accelerate.state import AcceleratorState

    # ファインチューニング実行
    run_training_task(...)

    # AcceleratorStateがクリアされていることを確認
    assert not AcceleratorState._shared_state
```

### 統合テスト
```python
# tests/test_finetuning_continual_sequence.py
def test_finetuning_then_continual_learning():
    """ファインチューニング→継続学習の順序実行テスト"""

    # 1. ファインチューニング実行
    result1 = run_training_task(task_id="ft_001", request=ft_request)
    assert result1.status == "completed"

    # 2. 継続学習実行（エラーが出ないことを確認）
    result2 = run_continual_learning_task(task_id="cl_001", request=cl_request)
    assert result2.status == "completed"
```

---

## 付録: AcceleratorState内部実装

```python
# accelerate/state.py (Accelerate v1.2.0)
class AcceleratorState:
    _shared_state = {}

    @staticmethod
    def _reset_state(reset_partial_state: bool = False):
        """Resets `_shared_state`, is used internally and should not be called"""
        AcceleratorState._shared_state.clear()
        if reset_partial_state:
            PartialState._reset_state()

    def __init__(self, mixed_precision=None, ...):
        self.__dict__ = self._shared_state
        if self._shared_state == {}:
            # 初回初期化
            self._shared_state['mixed_precision'] = mixed_precision
            # ...
        else:
            # 2回目以降：設定チェック
            if mixed_precision != self._shared_state['mixed_precision']:
                raise ValueError("AcceleratorState has already been initialized...")
```

---

## 結論

**根本原因**: LoRAファインチューニング終了後に`AcceleratorState._reset_state()`が呼ばれず、グローバル状態が残存

**推奨修正**: 解決策1（finally句でリセット）+ 解決策2（防御的実装）

**優先度**: 🔴 高（ユーザー体験に直接影響）

**所要時間**: 緊急対応30分、防御的実装25分、合計約1時間

**リスク**: 低（AcceleratorState._reset_state()は公式のリセットメソッド）

---

## 次のステップ

1. ⏳ Codex MCP検証結果の確認
2. ⏳ 解決策1の実装
3. ⏳ 単体テスト作成と実行
4. ⏳ 統合テスト実施
5. ⏳ 本番デプロイ

**Note**: Codex MCPの検証結果が出次第、このレポートを更新します。
