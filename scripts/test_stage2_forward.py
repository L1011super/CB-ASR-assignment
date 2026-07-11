"""Smoke test for stage-2 audio-Qwen forward pass."""

from __future__ import annotations

import argparse
import copy
import gc
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_stage2 import (
    build_candidate_builder,
    build_dataloader,
    build_model_and_tokenizers,
    load_config,
    print_trainable_parameters,
    resolve_manifest_path,
)


def run_once(config: dict, limit: int, with_hotwords: bool) -> None:
    cfg = copy.deepcopy(config)
    cfg["data"]["with_hotwords"] = with_hotwords
    if not with_hotwords:
        cfg["data"]["use_full_station_list"] = False
        cfg["data"]["num_distractors"] = 0
        cfg["data"]["max_prompt_length"] = min(int(cfg["data"]["max_prompt_length"]), 512)

    processor, tokenizer, model = build_model_and_tokenizers(cfg)
    print(f"\n=== with_hotwords={with_hotwords} ===")
    print_trainable_parameters(model)
    candidate_builder = build_candidate_builder(cfg)
    loader = build_dataloader(
        resolve_manifest_path(cfg["paths"]["train_manifest"]),
        cfg,
        processor,
        tokenizer,
        candidate_builder,
        limit=limit,
        shuffle=False,
    )
    batch = next(iter(loader))
    device = next(model.parameters()).device
    model.train()
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        outputs = model(
            input_features=batch["input_features"],
            prompt_input_ids=batch["prompt_input_ids"],
            prompt_attention_mask=batch["prompt_attention_mask"],
            answer_input_ids=batch["answer_input_ids"],
            answer_attention_mask=batch["answer_attention_mask"],
        )
    print("loss:", float(outputs["loss"].detach().cpu()))
    print("input_features:", tuple(batch["input_features"].shape))
    print("prompt_input_ids:", tuple(batch["prompt_input_ids"].shape))
    print("answer_input_ids:", tuple(batch["answer_input_ids"].shape))
    print("whisper_hidden:", outputs["whisper_hidden_shape"])
    print("speech_embeds:", outputs["speech_embeds_shape"])
    print("inputs_embeds:", outputs["inputs_embeds_shape"])
    print("target:", batch["target_stations"][0])
    print("prompt sample:", batch["prompts"][0][:500])
    del model, processor, tokenizer, loader, batch, outputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()

    config = load_config(args.config)
    run_once(config, limit=args.limit, with_hotwords=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    run_once(config, limit=args.limit, with_hotwords=False)


if __name__ == "__main__":
    main()
