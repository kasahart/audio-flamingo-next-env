# audio-flamingo-next

NVIDIA [Audio Flamingo Next](https://huggingface.co/nvidia/audio-flamingo-next-hf) の推論環境を DevContainer で構築するリポジトリです。  
NVIDIA GPU を使用した高速推論に対応しています。

Python 依存は requirements.txt ではなく pyproject.toml と uv.lock を正本として管理します。

Hugging Face のモデルキャッシュは /workspace/.cache/huggingface に保存するため、コンテナを作り直してもワーキングディレクトリが残っている限り再ダウンロードを避けられます。

## 必要条件

- [Docker](https://docs.docker.com/get-docker/)（GPU サポート付き）
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- CUDA 12.8 を利用できる NVIDIA ドライバー
- [VS Code](https://code.visualstudio.com/) + [Dev Containers 拡張機能](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

## クイックスタート

### 1. DevContainer を起動する

VS Code でこのリポジトリを開き、コマンドパレット（`Ctrl+Shift+P` / `Cmd+Shift+P`）から  
**「Dev Containers: Reopen in Container」** を選択します。

> コンテナのビルドには CUDA 12.8 ベースイメージと uv による依存解決を含むため、  
> 初回は数分かかることがあります。

### 2. GPU が認識されていることを確認する

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 3. 推論を実行する

```bash
uv run python inference.py --audio path/to/audio.wav --question "この音声は何ですか？"
```

### 4. 依存を更新する

```bash
uv lock
uv sync
```

## ファイル構成

```
.
├── .devcontainer/
│   ├── devcontainer.json   # DevContainer 設定（GPU アクセス含む）
│   └── Dockerfile          # CUDA 12.8 + uv + 非 root ユーザー設定
├── inference.py            # 推論サンプルスクリプト
├── pyproject.toml          # Python 依存定義
├── uv.lock                 # uv ロックファイル
├── requirements.txt        # 互換用の旧依存一覧
└── README.md
```

## 環境詳細

| 項目 | バージョン |
|------|------------|
| CUDA | 12.8.1 |
| cuDNN | 9.x |
| Python | 3.11 |
| PyTorch | 2.7+ (cu128) |
| パッケージ管理 | uv |
| transformers | Audio Flamingo Next 対応ビルド |

## 推論スクリプトのオプション

```
uv run python inference.py \
  --audio <音声ファイルパス>        # WAV / MP3 / FLAC など
  --question "質問テキスト"         # （省略時: "What do you hear in this audio?"）
  --model nvidia/audio-flamingo-next-hf  # HuggingFace モデル ID
  --device cuda                     # cuda または cpu
```

## モデルについて

- **モデル**: [nvidia/audio-flamingo-next-hf](https://huggingface.co/nvidia/audio-flamingo-next-hf)
- **タスク**: 音声理解・音声質問応答（AQA）
- **推奨 VRAM**: 24GB 以上（bfloat16 使用時）

## 注意事項

- Audio Flamingo Next は通常の `AutoModelForCausalLM` ではなく、モデルカードの推奨どおり `AutoModel` と chat template ベースで読み込みます。
- 一部の PyPI 版 `transformers` では `audioflamingonext` が未対応なため、このリポジトリでは uv の source 設定で Audio Flamingo Next 対応ビルドを固定しています。
- DevContainer は root ではなく vscode ユーザーで起動し、Python 仮想環境は /workspace/venv に作成されます。
- Hugging Face のキャッシュ先は /workspace/.cache/huggingface に固定してあり、ワークスペースが残る限りモデルキャッシュも維持されます。
