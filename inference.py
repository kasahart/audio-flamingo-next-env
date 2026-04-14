"""Audio Flamingo Next inference CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

DEFAULT_MODEL_ID = "nvidia/audio-flamingo-next-hf"
DEFAULT_QUESTION = "What do you hear in this audio?"
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Audio Flamingo Next Inference")
    parser.add_argument(
        "--audio",
        type=Path,
        required=True,
        help="Path to an audio file (WAV, MP3, FLAC, M4A, OGG)",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=DEFAULT_QUESTION,
        help="Question about the audio",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model ID",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to use for inference",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of generated tokens",
    )
    return parser.parse_args(argv)


def validate_runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    """Validate CLI inputs before model loading."""
    if not args.audio.exists():
        raise FileNotFoundError(f"Audio file not found: {args.audio}")

    if not args.audio.is_file():
        raise ValueError(f"Audio path is not a file: {args.audio}")

    if args.audio.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise ValueError(
            f"Unsupported audio extension: {args.audio.suffix or '<none>'}. "
            f"Supported extensions: {supported}"
        )

    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be a positive integer")

    args.audio = args.audio.resolve()
    return args


def ensure_runtime_dependencies() -> tuple[Any, tuple[Any, Any]]:
    """Import runtime-only dependencies lazily for easier testing and better errors."""
    try:
        import torch
        from transformers import AutoModel, AutoProcessor
    except ImportError as error:
        raise RuntimeError(
            "Missing runtime dependencies. Run `uv sync --frozen --no-dev` "
            "or install the packages from pyproject.toml."
        ) from error

    return torch, (AutoModel, AutoProcessor)


def resolve_device(requested_device: str) -> str:
    """Resolve the final device, falling back to CPU when CUDA is unavailable."""
    torch, _ = ensure_runtime_dependencies()

    if requested_device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Falling back to CPU.")
        return "cpu"

    return requested_device


def print_device_info(device: str) -> None:
    """Print runtime information for the selected device."""
    torch, _ = ensure_runtime_dependencies()

    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM: {total_memory:.1f} GB")


def load_model(model_id: str = DEFAULT_MODEL_ID, device: str = "cuda"):
    """Load the Audio Flamingo Next model and processor."""
    torch, transformer_classes = ensure_runtime_dependencies()
    auto_model, auto_processor = transformer_classes

    print(f"Loading model: {model_id}")
    model_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    processor = auto_processor.from_pretrained(model_id, trust_remote_code=True)

    try:
        model = auto_model.from_pretrained(
            model_id,
            dtype=model_dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
    except ValueError as error:
        if "audioflamingonext" in str(error).lower():
            raise RuntimeError(
                "The installed transformers build does not support Audio Flamingo Next. "
                "Install the repository dependencies again to get the supported build."
            ) from error
        raise

    if device != "cuda":
        model = model.to(device)

    model.eval()
    print(f"Model loaded successfully on {device}")
    return model, processor


def run_inference(
    audio_path: Path,
    question: str,
    model,
    processor,
    max_new_tokens: int = 512,
) -> str:
    """Run inference on an audio file with a text question."""
    torch, _ = ensure_runtime_dependencies()
    conversation = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "audio", "path": str(audio_path)},
                ],
            }
        ]
    ]

    batch = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    ).to(model.device)

    if "input_features" in batch:
        batch["input_features"] = batch["input_features"].to(model.dtype)

    with torch.inference_mode():
        generated_ids = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.2,
        )

    prompt_length = batch["input_ids"].shape[1]
    completion_ids = generated_ids[:, prompt_length:]
    output_text = processor.batch_decode(
        completion_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return output_text


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    try:
        args = validate_runtime_args(parse_args(argv))
        args.device = resolve_device(args.device)
        print_device_info(args.device)

        model, processor = load_model(model_id=args.model, device=args.device)
        answer = run_inference(
            audio_path=args.audio,
            question=args.question,
            model=model,
            processor=processor,
            max_new_tokens=args.max_new_tokens,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1

    print("\n--- Result ---")
    print(f"Question: {args.question}")
    print(f"Answer:   {answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
