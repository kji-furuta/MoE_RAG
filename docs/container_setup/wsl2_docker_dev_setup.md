# MoE_RAG WSL2 Docker 開発環境セットアップ

このガイドは Windows 上の WSL2 + Docker Desktop を用いて本リポジトリの開発環境を構築する手順をまとめたものです。GPU を利用する前提で説明していますが、GPU が無い場合でも基本的な API/フロントエンドは動作します。

## 1. 前提条件

- Windows 11 または Windows 10 22H2 以降
- BIOS/UEFI で仮想化機能 (Intel VT-x / AMD-V) と IOMMU を有効化済み
- Microsoft Store 版の Ubuntu (推奨: 22.04 LTS) を WSL2 でインストール済み
- 管理者権限で以下の機能を有効化
  ```powershell
  dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
  dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
  ```
- NVIDIA GPU 場合: [最新の Windows 用 GPU ドライバ](https://www.nvidia.co.jp/Download/index.aspx?lang=jp) と [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#wsl-2-backend) を導入
- Docker Desktop 4.30 以上 (設定画面で *Use the WSL 2 based engine* と対象ディストリビューションにチェック)
- Windows で `C:\Users\<username>\` 直下に書き込み権限があること (*~* 配下の `.wslconfig` を利用するため)

## 2. WSL2 全体設定

1. `docker/wslconfig.example` を参考に `C:\Users\<username>\.wslconfig` を作成し、メモリ/CPU/スワップを調整します。大規模モデルを扱う場合は 32GB 以上のメモリ割り当てを推奨します。
2. 設定を反映させるために WSL を終了
   ```powershell
   wsl --shutdown
   ```

## 3. Ubuntu (WSL) 側の初期設定

WSL のシェルで以下を実行して基本ツールを揃えます。

```bash
sudo apt update
sudo apt install -y build-essential git curl python3 python3-venv pkg-config
```

GPU を使う場合は追加で

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \\
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \\
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \\
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo service docker restart
```

## 4. リポジトリの配置

WSL の ext4 ファイルシステム (`/home/<user>` や `/mnt/wsl`) に配置することでファイル監視と I/O が高速になります。

```bash
cd ~
git clone https://github.com/<your-organization>/MoE_RAG.git
cd MoE_RAG
```

## 5. 環境変数とローカルディレクトリ準備

```bash
cp docker/.env.example docker/.env
```

`docker/.env` を開き、以下の値を必要に応じて設定します。

- `WANDB_API_KEY` (Weights & Biases を利用する場合)
- `HF_TOKEN` (Hugging Face の gated モデルを利用する場合)
- `JUPYTER_TOKEN` (Jupyter Lab のアクセス制御)
- 必要であれば `OPENAI_API_KEY` などの追加キー

データやモデルの永続化用ディレクトリが無い場合は作成します。

```bash
mkdir -p data models outputs temp_uploads qdrant_data metadata
```

## 6. 開発コンテナの起動

Docker Desktop を起動した状態で、WSL シェルから以下を実行します。

```bash
# 初回はイメージのビルドを伴うため時間がかかります
./start_dev_env.sh
```

スクリプトは次を自動的に行います。

- WSL2 上であることを検出してメッセージを表示
- `docker compose` プラグインまたは `docker-compose` コマンドを自動判別
- `docker/docker-compose.yml` を用いてイメージのビルドとコンテナ起動
- `ai-ft-container` と `ai-ft-qdrant` の稼働チェック
- FastAPI (port 8050) のヘルスチェック

起動後にアクセス可能な主要エンドポイント

- Web UI: http://localhost:8050
- RAG API: http://localhost:8050/rag
- Qdrant UI: http://localhost:6333/dashboard
- Jupyter Lab: http://localhost:8888 (トークンは `.env` の値)
- TensorBoard: http://localhost:6006

## 7. 動作確認

```bash
# GPU 利用状況を確認
docker exec -it ai-ft-container python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"

# RAG API の疎通確認
curl -X POST http://localhost:8050/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "設計速度80km/hの道路の最小曲線半径は？", "top_k": 3}'
```

## 8. 停止とクリーンアップ

```bash
./stop_dev_env.sh          # コンテナの停止

# データを含めて完全に削除したい場合 (要注意)
cd docker
docker compose down -v
```

## 9. トラブルシュート

- **Docker Desktop が WSL2 を検出しない**: Docker Desktop の `Settings > Resources > WSL Integration` で対象ディストリビューションのチェックを確認。
- **GPU がコンテナから見えない**: `nvidia-smi` を Windows で実行してハードウェア認識を確認し、`nvidia-container-toolkit` の設定を見直す。
- **ポート競合**: 既に `8050` 等を使用しているプロセスがある場合は `docker/docker-compose.yml` のポートマッピングを変更。
- **ファイル監視が反映されない**: プロジェクトを `/mnt/c/` 下ではなく WSL のネイティブパスに置く。

これで WSL2 + Docker ベースの MoE_RAG 開発環境が整います。`docker/docker-compose.yml` のボリュームマウントによりホスト側のソースをリアルタイムに編集しながら、コンテナ内で推論・学習を実行できます。
