"""Smoke test for stage-1 dataloader and model forward."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_stage1 import (  # noqa: E402
    build_dataloader,
    build_model_and_tokenizers,
    load_config,
    print_trainable_parameters,
    resolve_path,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    processor, tokenizer, model = build_model_and_tokenizers(config)
    print_trainable_parameters(model)
    loader = build_dataloader(
        resolve_path(config["paths"]["train_manifest"]),
        config,
        processor,
        tokenizer,
        limit=args.limit,
        shuffle=False,
    )

    batch = next(iter(loader))
    device = next(model.parameters()).device
    fp16 = bool(config["training"].get("fp16", True))
    model.train()
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda" and fp16):
        outputs = model(
            input_features=batch["input_features"],
            prompt_input_ids=batch["prompt_input_ids"],
            prompt_attention_mask=batch["prompt_attention_mask"],
            answer_input_ids=batch["answer_input_ids"],
            answer_attention_mask=batch["answer_attention_mask"],
        )

    result = {
        "loss": float(outputs["loss"].detach().cpu()),
        "input_features_shape": tuple(batch["input_features"].shape),
        "prompt_input_ids_shape": tuple(batch["prompt_input_ids"].shape),
        "answer_input_ids_shape": tuple(batch["answer_input_ids"].shape),
        "whisper_hidden_shape": outputs["whisper_hidden_shape"],
        "speech_embeds_shape": outputs["speech_embeds_shape"],
        "inputs_embeds_shape": outputs["inputs_embeds_shape"],
        "utt_ids": batch["utt_ids"],
        "target_stations": batch["target_stations"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
