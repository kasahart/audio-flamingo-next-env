"""
Audio Flamingo Next - Inference Example
Model: nvidia/audio-flamingo-next-hf

Usage:
    python inference.py --audio path/to/audio.wav --question "What sound is this?"
"""

import argparse
import torch
from transformers import AutoModel, AutoProcessor


def load_model(model_id: str = "nvidia/audio-flamingo-next-hf", device: str = "cuda"):
    """Load the Audio Flamingo Next model and processor."""
    print(f"Loading model: {model_id}")
    print(f"Device: {device}")

    model_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    try:
        model = AutoModel.from_pretrained(
            model_id,
            dtype=model_dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
    except ValueError as error:
        if "audioflamingonext" in str(error).lower():
            raise RuntimeError(
                "The installed transformers build does not support Audio Flamingo Next. "
                "Install the repository requirements again to get the supported build."
            ) from error
        raise

    if device != "cuda":
        model = model.to(device)

    model.eval()

    print(f"Model loaded successfully on {device}")
    return model, processor


def run_inference(audio_path: str, question: str, model, processor, device: str = "cuda") -> str:
    """Run inference on an audio file with a text question."""
    conversation = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "audio", "path": audio_path},
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

    # Generate response
    with torch.inference_mode():
        generated_ids = model.generate(
            **batch,
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.2,
        )

    # Decode output
    prompt_length = batch["input_ids"].shape[1]
    completion_ids = generated_ids[:, prompt_length:]
    output_text = processor.batch_decode(
        completion_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return output_text


def main():
    parser = argparse.ArgumentParser(description="Audio Flamingo Next Inference")
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to audio file (WAV, MP3, FLAC, etc.)",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="What do you hear in this audio?",
        help="Question about the audio",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/audio-flamingo-next-hf",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference (cuda/cpu)",
    )
    args = parser.parse_args()

    # Check GPU availability
    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("WARNING: CUDA is not available. Falling back to CPU.")
            args.device = "cpu"
        else:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load model and run inference
    model, processor = load_model(model_id=args.model, device=args.device)
    answer = run_inference(
        audio_path=args.audio,
        question=args.question,
        model=model,
        processor=processor,
        device=args.device,
    )

    print("\n--- Result ---")
    print(f"Question: {args.question}")
    print(f"Answer:   {answer}")


if __name__ == "__main__":
    main()
