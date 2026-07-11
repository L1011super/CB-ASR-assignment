"""Single-file inference for real-person wav tests.

Usage:
    python real_people_test/real_people_test.py lsdtest.wav

The input wav is resolved under:
    real_people_test/inpt/<wav_name>

The script loads the best trained stage-2 model and prints the predicted
standard Tianjin metro station name, or null if no legal station is predicted.
"""

from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.infer_manifest import (  # noqa: E402
    build_model,
    load_yaml,
    parse_prediction,
    resolve_checkpoint_dir,
)
from src.data.audio_dataset import AudioSample  # noqa: E402
from src.data.collator import Stage2AudioQwenCollator  # noqa: E402
from src.data.prompts import StationCandidateBuilder  # noqa: E402
from src.data.station_lexicon import load_station_lexicon, station_names  # noqa: E402
from src.training.train_stage2 import resolve_path  # noqa: E402

DEFAULT_CKPT = PROJECT_ROOT / "outputs/stage2_bias_hotwords_attention_q32_hard_refine/best"
INPUT_DIR = Path(__file__).resolve().parent / "inpt"


@lru_cache(maxsize=1)
def _load_inference_bundle(ckpt: str, with_hotwords: bool) -> dict[str, Any]:
    """Load model/tokenizer/processor once per process."""

    ckpt_dir = resolve_checkpoint_dir(ckpt)
    config = load_yaml(ckpt_dir / "training_config.yaml")
    config["data"]["with_hotwords"] = with_hotwords
    if with_hotwords:
        config["data"]["use_full_station_list"] = True
        config["data"]["num_distractors"] = None
        config["data"]["max_prompt_length"] = max(
            int(config["data"].get("max_prompt_length", 2048)),
            2048,
        )

    processor, tokenizer, model = build_model(ckpt_dir, config)
    lexicon = load_station_lexicon(resolve_path(config["paths"]["station_csv"]))
    names = station_names(lexicon)
    candidate_builder = StationCandidateBuilder(
        station_names=names,
        num_distractors=None,
        use_full_list=True,
        seed=int(config.get("seed", 42)),
    )
    collator = Stage2AudioQwenCollator(
        whisper_processor=processor,
        qwen_tokenizer=tokenizer,
        candidate_builder=candidate_builder,
        with_hotwords=with_hotwords,
        target_sample_rate=int(config["data"]["target_sample_rate"]),
        max_prompt_length=int(config["data"]["max_prompt_length"]),
        max_answer_length=int(config["data"]["max_answer_length"]),
    )
    return {
        "config": config,
        "tokenizer": tokenizer,
        "model": model,
        "valid_names": set(names),
        "collator": collator,
    }


def _resolve_input_wav(wav_name: str) -> Path:
    """Resolve user input to real_people_test/inpt/<name>.wav."""

    name = Path(wav_name).name
    if not name.lower().endswith(".wav"):
        name = f"{name}.wav"
    audio_path = INPUT_DIR / name
    if not audio_path.exists():
        raise FileNotFoundError(f"Input wav not found: {audio_path}")
    return audio_path


def infer_wav(
    wav_name: str,
    ckpt: str | Path = DEFAULT_CKPT,
    with_hotwords: bool = True,
    print_raw: bool = False,
) -> str | None:
    """Infer station name for real_people_test/inpt/<wav_name>.

    Args:
        wav_name: File name such as ``"lsdtest.wav"``. The file is read from
            ``real_people_test/inpt``.
        ckpt: Stage-2 checkpoint directory. Defaults to the best hard-refine
            checkpoint from this project.
        with_hotwords: Whether to include the full station hotword list.
        print_raw: If true, print the model's raw JSON/text output as well.

    Returns:
        The predicted standard station name, or ``None`` for null.
    """

    audio_path = _resolve_input_wav(wav_name)
    bundle = _load_inference_bundle(str(Path(ckpt)), bool(with_hotwords))
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    config = bundle["config"]
    collator = bundle["collator"]
    valid_names = bundle["valid_names"]

    sample = AudioSample(
        utt_id=audio_path.stem,
        audio_path=audio_path,
        text=None,
        target_station=None,
        has_station=False,
        raw={"audio_path": audio_path.as_posix()},
    )
    loader = DataLoader([sample], batch_size=1, shuffle=False, num_workers=0, collate_fn=collator)
    batch = next(iter(loader))

    device = next(model.parameters()).device
    fp16 = device.type == "cuda" and bool(config["training"].get("fp16", True))
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
        generated = model.generate(
            input_features=batch["input_features"],
            prompt_input_ids=batch["prompt_input_ids"],
            prompt_attention_mask=batch["prompt_attention_mask"],
            max_new_tokens=32,
            do_sample=False,
            temperature=None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    raw_output = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
    pred_station = parse_prediction(raw_output, valid_names)
    if print_raw:
        print(f"raw_output: {raw_output}")
    print(pred_station if pred_station is not None else "null")
    return pred_station


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer station from real_people_test/inpt/<wav>.")
    parser.add_argument("wav_name", help="wav file name under real_people_test/inpt, e.g. lsdtest.wav")
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT), help="stage-2 checkpoint directory")
    parser.add_argument("--no_hotwords", action="store_true", help="disable station hotword list")
    parser.add_argument("--print_raw", action="store_true", help="print raw model output")
    args = parser.parse_args()
    infer_wav(
        wav_name=args.wav_name,
        ckpt=args.ckpt,
        with_hotwords=not args.no_hotwords,
        print_raw=args.print_raw,
    )


if __name__ == "__main__":
    main()
