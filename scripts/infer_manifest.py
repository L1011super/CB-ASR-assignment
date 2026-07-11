"""Run stage-2 checkpoint inference on a JSONL manifest."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.audio_dataset import AudioManifestDataset
from src.data.collator import Stage2AudioQwenCollator
from src.data.prompts import StationCandidateBuilder
from src.data.station_lexicon import load_station_lexicon, station_names
from src.models.audio_qwen import AudioQwenForStage2
from src.training.train_stage2 import resolve_manifest_path, resolve_path


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = value.strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def resolve_checkpoint_dir(path: str | Path) -> Path:
    ckpt = resolve_path(path)
    if (ckpt / "best").exists():
        ckpt = ckpt / "best"
    required = [ckpt / "adapter_projector.pt", ckpt / "lora_adapter", ckpt / "training_config.yaml"]
    missing = [item for item in required if not item.exists()]
    if missing:
        raise FileNotFoundError(f"Invalid stage-2 checkpoint, missing: {missing}")
    return ckpt


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(ckpt_dir: Path, config: dict[str, Any]) -> tuple[Any, Any, AudioQwenForStage2]:
    whisper_path = resolve_path(config["paths"]["whisper_model"])
    qwen_path = resolve_path(config["paths"]["qwen_model"])
    processor = AutoProcessor.from_pretrained(str(whisper_path), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(qwen_path), local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    requested_device = str(config["training"].get("device", "cuda:0"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {requested_device}, but CUDA is not available.")
    device = torch.device(requested_device)
    dtype = torch.float16 if device.type == "cuda" and bool(config["training"].get("fp16", True)) else torch.float32
    model_cfg = config["model"]
    model = AudioQwenForStage2(
        whisper_model_path=whisper_path,
        qwen_model_path=qwen_path,
        adapter_type=str(model_cfg["adapter_type"]),
        num_query_tokens=int(model_cfg["num_query_tokens"]),
        adapter_hidden_size=int(model_cfg["adapter_hidden_size"]),
        adapter_dropout=float(model_cfg["adapter_dropout"]),
        freeze_whisper=True,
        torch_dtype=dtype,
    )
    model.load_adapter_projector(ckpt_dir / "adapter_projector.pt")
    model.load_lora(ckpt_dir / "lora_adapter")
    model.to(device)
    model.eval()
    return processor, tokenizer, model


def build_loader(
    manifest: Path,
    config: dict[str, Any],
    processor: Any,
    tokenizer: Any,
    with_hotwords: bool,
    limit: int | None,
) -> DataLoader:
    lexicon = load_station_lexicon(resolve_path(config["paths"]["station_csv"]))
    candidates = StationCandidateBuilder(
        station_names=station_names(lexicon),
        num_distractors=None,
        use_full_list=True,
        seed=int(config.get("seed", 42)),
    )
    dataset = AudioManifestDataset(
        manifest_path=manifest,
        repo_root=PROJECT_ROOT,
        use_null_samples=True,
        limit=limit,
    )
    collator = Stage2AudioQwenCollator(
        whisper_processor=processor,
        qwen_tokenizer=tokenizer,
        candidate_builder=candidates,
        with_hotwords=with_hotwords,
        target_sample_rate=int(config["data"]["target_sample_rate"]),
        max_prompt_length=int(config["data"]["max_prompt_length"]),
        max_answer_length=int(config["data"]["max_answer_length"]),
    )
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collator)


def parse_prediction(raw_output: str, valid_names: set[str]) -> str | None:
    text = raw_output.strip()
    json_match = re.search(r"\{.*?\}", text, flags=re.S)
    candidates = [text]
    if json_match:
        candidates.insert(0, json_match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        station = value.get("station") if isinstance(value, dict) else None
        if station is None:
            return None
        station_text = str(station).strip()
        return station_text if station_text in valid_names else None
    for name in sorted(valid_names, key=len, reverse=True):
        if name and name in text:
            return name
    if "null" in text.lower():
        return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--with_hotwords", type=parse_bool, default=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    ckpt_dir = resolve_checkpoint_dir(args.ckpt)
    config = load_yaml(ckpt_dir / "training_config.yaml")
    config["data"]["with_hotwords"] = bool(args.with_hotwords)
    if args.with_hotwords:
        config["data"]["use_full_station_list"] = True
        config["data"]["num_distractors"] = None
        config["data"]["max_prompt_length"] = max(int(config["data"].get("max_prompt_length", 2048)), 2048)

    processor, tokenizer, model = build_model(ckpt_dir, config)
    manifest = resolve_manifest_path(args.manifest)
    loader = build_loader(manifest, config, processor, tokenizer, bool(args.with_hotwords), args.limit)
    valid_names = set(station_names(load_station_lexicon(resolve_path(config["paths"]["station_csv"]))))

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    fp16 = device.type == "cuda" and bool(config["training"].get("fp16", True))

    with output_path.open("w", encoding="utf-8") as f:
        for batch in tqdm(loader, desc="stage2 infer"):
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
            row = {
                "utt_id": batch["utt_ids"][0],
                "target_station": batch["target_stations"][0],
                "raw_output": raw_output,
                "pred_station": pred_station,
                "with_hotwords": bool(args.with_hotwords),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
