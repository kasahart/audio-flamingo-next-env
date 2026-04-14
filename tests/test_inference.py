import argparse
import tempfile
import unittest
from pathlib import Path

import inference


class ValidateRuntimeArgsTest(unittest.TestCase):
    def test_parse_args_uses_defaults(self):
        args = inference.parse_args(["--audio", "sample.wav"])

        self.assertEqual(args.audio, Path("sample.wav"))
        self.assertEqual(args.question, inference.DEFAULT_QUESTION)
        self.assertEqual(args.model, inference.DEFAULT_MODEL_ID)
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.max_new_tokens, 512)

    def test_validate_runtime_args_accepts_supported_audio_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "sample.wav"
            audio_path.write_bytes(b"RIFF")
            args = argparse.Namespace(
                audio=audio_path,
                question="What is this?",
                model=inference.DEFAULT_MODEL_ID,
                device="cpu",
                max_new_tokens=64,
            )

            validated = inference.validate_runtime_args(args)

            self.assertEqual(validated.audio, audio_path.resolve())

    def test_validate_runtime_args_rejects_missing_file(self):
        args = argparse.Namespace(
            audio=Path("missing.wav"),
            question="What is this?",
            model=inference.DEFAULT_MODEL_ID,
            device="cpu",
            max_new_tokens=64,
        )

        with self.assertRaises(FileNotFoundError):
            inference.validate_runtime_args(args)

    def test_validate_runtime_args_rejects_unsupported_extension(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "sample.txt"
            audio_path.write_text("not audio", encoding="utf-8")
            args = argparse.Namespace(
                audio=audio_path,
                question="What is this?",
                model=inference.DEFAULT_MODEL_ID,
                device="cpu",
                max_new_tokens=64,
            )

            with self.assertRaises(ValueError):
                inference.validate_runtime_args(args)

    def test_validate_runtime_args_rejects_non_positive_max_new_tokens(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "sample.wav"
            audio_path.write_bytes(b"RIFF")
            args = argparse.Namespace(
                audio=audio_path,
                question="What is this?",
                model=inference.DEFAULT_MODEL_ID,
                device="cpu",
                max_new_tokens=0,
            )

            with self.assertRaises(ValueError):
                inference.validate_runtime_args(args)


if __name__ == "__main__":
    unittest.main()