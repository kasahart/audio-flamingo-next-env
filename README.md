# audio-flamingo-next

NVIDIA [Audio Flamingo Next](https://huggingface.co/nvidia/audio-flamingo-next-hf) の推論環境を DevContainer で構築するリポジトリです。  
NVIDIA RTX PRO 6000 Blackwell GPU を使用した高速推論に対応しています。

## 必要条件

- [Docker](https://docs.docker.com/get-docker/)（GPU サポート付き）
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NVIDIA ドライバー 570 以上（Blackwell GPU 対応）
- [VS Code](https://code.visualstudio.com/) + [Dev Containers 拡張機能](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

## クイックスタート

### 1. DevContainer を起動する

VS Code でこのリポジトリを開き、コマンドパレット（`Ctrl+Shift+P` / `Cmd+Shift+P`）から  
**「Dev Containers: Reopen in Container」** を選択します。

> コンテナのビルドには CUDA 12.8 ベースイメージと PyTorch のダウンロードを含むため、  
> 初回は数分かかることがあります。

### 2. GPU が認識されていることを確認する

```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 3. 推論を実行する

```bash
python3 inference.py --audio path/to/audio.wav --question "この音声は何ですか？"
```

## ファイル構成

```
.
├── .devcontainer/
│   ├── devcontainer.json   # DevContainer 設定（GPU アクセス含む）
│   └── Dockerfile          # CUDA 12.8 + PyTorch 2.7 + 依存ライブラリ
├── inference.py            # 推論サンプルスクリプト
├── requirements.txt        # Python 依存パッケージ
└── README.md
```

## 環境詳細

| 項目 | バージョン |
|------|------------|
| CUDA | 12.8.1 |
| cuDNN | 9.x |
| Python | 3.11 |
| PyTorch | 2.7+ (cu128) |
| transformers | ≥ 4.51.0 |
| GPU アーキテクチャ | Blackwell (sm_120) |

## 推論スクリプトのオプション

```
python3 inference.py \
  --audio <音声ファイルパス>        # WAV / MP3 / FLAC など
  --question "質問テキスト"         # （省略時: "What do you hear in this audio?"）
  --model nvidia/audio-flamingo-next-hf  # HuggingFace モデル ID
  --device cuda                     # cuda または cpu
```

## モデルについて

- **モデル**: [nvidia/audio-flamingo-next-hf](https://huggingface.co/nvidia/audio-flamingo-next-hf)
- **タスク**: 音声理解・音声質問応答（AQA）
- **推奨 VRAM**: 24GB 以上（bfloat16 使用時）
