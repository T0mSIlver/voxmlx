__version__ = "0.0.2"

import argparse
from pathlib import Path

from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy
from mistral_common.tokens.tokenizers.tekken import Tekkenizer

from .generate import generate
from .weights import download_model, load_model as _load_weights


def _load_tokenizer(model_path: Path) -> Tekkenizer:
    tekken_path = model_path / "tekken.json"
    return Tekkenizer.from_file(str(tekken_path))


def _build_prompt_tokens(
    sp: Tekkenizer,
    n_left_pad_tokens: int = 32,
    num_delay_tokens: int = 6,
) -> tuple[list[int], int]:
    streaming_pad = sp.get_special_token("[STREAMING_PAD]")
    prefix_len = n_left_pad_tokens + num_delay_tokens  # 38 STREAMING_PAD tokens
    tokens = [sp.bos_id] + [streaming_pad] * prefix_len
    return tokens, num_delay_tokens


def load_model(model_path: str = "mlx-community/Voxtral-Mini-4B-Realtime-6bit"):
    if not Path(model_path).exists():
        model_path = download_model(model_path)
    else:
        model_path = Path(model_path)

    model, config = _load_weights(model_path)
    sp = _load_tokenizer(model_path)
    return model, sp, config


def transcribe(
    audio_path: str,
    model_path: str = "mlx-community/Voxtral-Mini-4B-Realtime-6bit",
    temperature: float = 0.0,
) -> str:
    model, sp, config = load_model(model_path)

    prompt_tokens, n_delay_tokens = _build_prompt_tokens(sp)

    output_tokens = generate(
        model,
        audio_path,
        prompt_tokens,
        n_delay_tokens=n_delay_tokens,
        temperature=temperature,
        eos_token_id=sp.eos_id,
    )

    return sp.decode(output_tokens, special_token_policy=SpecialTokenPolicy.IGNORE)


def main():
    parser = argparse.ArgumentParser(description="Voxtral Mini Realtime speech-to-text")
    parser.add_argument("--audio", default=None, help="Path to audio file (omit to stream from mic)")
    parser.add_argument("--model", default="mlx-community/Voxtral-Mini-4B-Realtime-6bit", help="Model path or HF model ID")
    parser.add_argument("--temp", type=float, default=0.0, help="Sampling temperature (0 = greedy)")
    parser.add_argument("--serve", action="store_true", help="Start WebSocket server (OpenAI Realtime API compatible)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (used with --serve)")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (used with --serve)")
    args = parser.parse_args()

    if args.serve:
        from .server import create_app
        import logging
        import uvicorn

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
        )
        app = create_app(args.model, args.temp)
        uvicorn.run(app, host=args.host, port=args.port)
    elif args.audio is not None:
        text = transcribe(
            args.audio,
            model_path=args.model,
            temperature=args.temp,
        )
        print(text)
    else:
        from .stream import stream_transcribe

        stream_transcribe(
            model_path=args.model,
            temperature=args.temp,
        )
