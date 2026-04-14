"""
Audio Flamingo Next - Inference Example
Model: nvidia/audio-flamingo-next-hf
GPU: NVIDIA RTX PRO 6000 Blackwell

Usage:
    python inference.py --audio path/to/audio.wav --question "What sound is this?"
"""

import argparse
import torch
from transformers import AutoProcessor, AutoModelForCausalLM


def load_model(model_id: str = "nvidia/audio-flamingo-next-hf", device: str = "cuda"):
    """Load the Audio Flamingo Next model and processor."""
    print(f"Loading model: {model_id}")
    print(f"Device: {device}")

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    print(f"Model loaded successfully on {device}")
    return model, processor


def run_inference(audio_path: str, question: str, model, processor, device: str = "cuda") -> str:
    """Run inference on an audio file with a text question."""
    import librosa

    # Load audio
    audio_array, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
    print(f"Audio loaded: {audio_path} ({len(audio_array)/sample_rate:.2f}s at {sample_rate}Hz)")

    # Prepare inputs
    inputs = processor(
        text=question,
        audio=audio_array,
        sampling_rate=sample_rate,
        return_tensors="pt",
    ).to(device)

    # Generate response
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

    # Decode output
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
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
