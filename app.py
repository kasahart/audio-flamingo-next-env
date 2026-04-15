"""Streamlit UI for Audio Flamingo Next inference.

Loads the model once per session using st.cache_resource,
so weights stay resident across multiple requests.

Usage:
    uv run streamlit run app.py
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import soundfile as sf
import streamlit as st
import wandas

import inference as _inf

matplotlib.use("Agg")  # non-interactive backend for server use

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Audio Flamingo Next",
    page_icon="🦩",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Model loading – cached for the lifetime of the Streamlit server process
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="モデルをロード中…（初回のみ）")
def load_model(model_id: str, device: str):
    device = _inf.resolve_device(device)
    model, processor = _inf.load_model(model_id=model_id, device=device)
    return model, processor, device


# ---------------------------------------------------------------------------
# Helper: build wandas ChannelFrame from uploaded bytes
# ---------------------------------------------------------------------------
def _to_channel_frame(data: bytes, filename: str) -> wandas.ChannelFrame:
    """Load audio bytes into a wandas ChannelFrame.

    WAV files are passed directly to wandas.read_wav().
    Other formats are decoded with soundfile first.
    The original filename is preserved as the frame label.
    """
    suffix = Path(filename).suffix.lower()
    stem = Path(filename).stem

    if suffix == ".wav":
        buf = io.BytesIO(data)
        buf.name = filename  # wandas uses .name to set the frame label
        return wandas.read_wav(buf)

    # Decode non-WAV formats with soundfile, then wrap in ChannelFrame
    audio_np, sr = sf.read(io.BytesIO(data), always_2d=True, dtype="float32")
    # soundfile returns (samples, channels) → wandas expects (channels, samples)
    return wandas.from_ndarray(audio_np.T, sampling_rate=sr, frame_label=stem)


# ---------------------------------------------------------------------------
# Sidebar – settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("設定")
    model_id = st.text_input("モデル ID", value=_inf.DEFAULT_MODEL_ID)
    device_choice = st.selectbox("デバイス", ["cuda", "cpu"], index=0)
    max_new_tokens = st.slider("最大生成トークン数", min_value=64, max_value=1024, value=512, step=64)
    st.divider()
    st.subheader("可視化オプション")
    fmin = st.number_input("最低周波数 fmin (Hz)", min_value=0, max_value=20000, value=0, step=100)
    fmax_input = st.number_input("最高周波数 fmax (Hz, 0=自動)", min_value=0, max_value=20000, value=0, step=100)
    fmax: float | None = float(fmax_input) if fmax_input > 0 else None
    cmap = st.selectbox("カラーマップ", ["jet", "viridis", "magma", "inferno", "plasma"], index=0)
    apply_aw = st.checkbox("A特性補正 (Aw)", value=False)
    st.divider()
    st.caption("モデルは初回リクエスト時に一度だけロードされ、以降はメモリに常駐します。")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("🦩 Audio Flamingo Next")

uploaded = st.file_uploader(
    "音声ファイルをアップロード（WAV / MP3 / FLAC / M4A / OGG）",
    type=["wav", "mp3", "flac", "m4a", "ogg"],
)

# ---------------------------------------------------------------------------
# Audio preview – show as soon as a file is uploaded
# ---------------------------------------------------------------------------
if uploaded is not None:
    audio_bytes = uploaded.read()
    uploaded.seek(0)  # reset for later use

    st.subheader("プレビュー")
    st.audio(audio_bytes, format=f"audio/{Path(uploaded.name).suffix.lstrip('.')}")

    # wandas describe: waveform + spectrogram + Welch spectrum
    with st.spinner("波形・スペクトログラムを描画中…"):
        try:
            cf = _to_channel_frame(audio_bytes, uploaded.name)
            # describe 内部で ax.set(ylim=None) が呼ばれると specshow の
            # fmin/fmax 設定がリセットされるため、ylim を明示的に渡す。
            ylim: tuple | None = (
                (float(fmin), fmax) if (fmin > 0 or fmax is not None) else None
            )
            figs: list = cf.describe(
                is_close=False,
                fmin=float(fmin),
                fmax=fmax,
                ylim=ylim,
                cmap=cmap,
                Aw=apply_aw,
            )
            for i, fig in enumerate(figs):
                ch_label = f"チャンネル {i + 1}" if len(figs) > 1 else ""
                if ch_label:
                    st.caption(ch_label)
                st.pyplot(fig)
                plt.close(fig)
        except Exception as exc:
            st.warning(f"可視化をスキップしました: {exc}")

    st.divider()

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
st.subheader("推論")
question = st.text_input(
    "質問",
    value=_inf.DEFAULT_QUESTION,
    placeholder="例: この音声は何ですか？",
)

run = st.button("推論を実行", type="primary", disabled=uploaded is None)

if run and uploaded is not None:
    suffix = Path(uploaded.name).suffix.lower()
    uploaded.seek(0)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = Path(tmp.name)

    try:
        model, processor, device = load_model(model_id, device_choice)

        with st.spinner("推論中…"):
            answer = _inf.run_inference(
                audio_path=tmp_path,
                question=question,
                model=model,
                processor=processor,
                max_new_tokens=max_new_tokens,
            )

        st.success("完了")
        st.subheader("回答")
        st.write(answer)

        with st.expander("詳細"):
            st.write(f"**モデル**: {model_id}")
            st.write(f"**デバイス**: {device}")
            st.write(f"**ファイル**: {uploaded.name}")
            st.write(f"**質問**: {question}")
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        st.error(f"エラー: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)
