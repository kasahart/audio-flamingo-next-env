# audio-flamingo-next

NVIDIA [Audio Flamingo Next](https://huggingface.co/nvidia/audio-flamingo-next-hf) の推論環境を DevContainer で構築するリポジトリです。  
NVIDIA GPU を使用した高速推論に対応しています。
再現性のある DevContainer 構成と uv ベースの依存管理を中心に整理しています。

Python 依存は requirements.txt ではなく pyproject.toml と uv.lock を正本として管理します。

Hugging Face のモデルキャッシュは /workspace/.cache/huggingface に保存するため、コンテナを作り直してもワーキングディレクトリが残っている限り再ダウンロードを避けられます。

## このリポジトリでできること

- NVIDIA Audio Flamingo Next の単発推論
- GPU 対応 DevContainer 上での再現可能な実行環境構築

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

### 5. 必要に応じて Hugging Face 認証を設定する

モデル取得時のレート制限や警告を避けたい場合は、事前に `HF_TOKEN` を設定してください。

```bash
export HF_TOKEN=your_huggingface_token
```

### 6. 軽量チェックを実行する

```bash
python -m unittest discover -s tests -v
```

## ファイル構成

```
.
├── .devcontainer/
│   ├── devcontainer.json   # DevContainer 設定（GPU アクセス含む）
│   └── Dockerfile          # CUDA 12.8 + uv + 非 root ユーザー設定
├── inference.py            # 推論サンプルスクリプト
├── pyproject.toml          # Python 依存定義
├── tests/                  # 軽量ユニットテスト
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
  --max-new-tokens 512              # 生成トークン数上限
```

存在しないファイルや未対応拡張子を指定した場合は、モデルをロードする前に明示的にエラー終了します。

## モデルについて

- **モデル**: [nvidia/audio-flamingo-next-hf](https://huggingface.co/nvidia/audio-flamingo-next-hf)
- **タスク**: 音声理解・音声質問応答（AQA）
- **推奨 VRAM**: 24GB 以上（bfloat16 使用時）

## 注意事項

- Audio Flamingo Next は通常の `AutoModelForCausalLM` ではなく、モデルカードの推奨どおり `AutoModel` と chat template ベースで読み込みます。
- 一部の PyPI 版 `transformers` では `audioflamingonext` が未対応なため、このリポジトリでは uv の source 設定で Audio Flamingo Next 対応ビルドを固定しています。
- DevContainer は root ではなく vscode ユーザーで起動し、Python 仮想環境は /workspace/venv に作成されます。
- Hugging Face のキャッシュ先は /workspace/.cache/huggingface に固定してあり、ワークスペースが残る限りモデルキャッシュも維持されます。

## トラブルシューティング
  ｗ
- `ERROR: Audio file not found`:
  指定した音声ファイルのパスを確認してください。
- `WARNING: CUDA is not available. Falling back to CPU.`:
  GPU が見えていないため CPU 実行に切り替わっています。DevContainer を GPU 付きで起動しているか確認してください。
- `The installed transformers build does not support Audio Flamingo Next`:
  `uv sync --frozen --no-dev` を再実行し、README 記載の依存構成を使ってください。
- Hugging Face の未認証アクセス警告が出る:
  必要に応じて `HF_TOKEN` を設定してください。

## ライセンス

MIT License を採用しています。詳細は LICENSE を参照してください。
