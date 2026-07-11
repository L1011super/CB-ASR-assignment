"""Stage-2 contextual-biasing training with station hotwords and LoRA."""

from __future__ import annotations

import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, get_linear_schedule_with_warmup

from src.data.audio_dataset import AudioManifestDataset
from src.data.collator import Stage2AudioQwenCollator
from src.data.prompts import StationCandidateBuilder
from src.data.station_lexicon import load_station_lexicon, station_names
from src.models.audio_qwen import AudioQwenForStage2, count_parameters, trainable_parameter_report
from src.models.lora_utils import load_lora_adapter, lora_target_report


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def resolve_manifest_path(path: str | Path) -> Path:
    resolved = resolve_path(path)
    if resolved.exists():
        return resolved
    if resolved.name in {"train.jsonl", "valid.jsonl", "test.jsonl"}:
        fallback = resolved.with_name(resolved.stem + "_tts.jsonl")
        if fallback.exists():
            return fallback
    return resolved


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_candidate_builder(config: dict[str, Any]) -> StationCandidateBuilder:
    data_config = config["data"]
    lexicon = load_station_lexicon(resolve_path(config["paths"]["station_csv"]))
    return StationCandidateBuilder(
        station_names=station_names(lexicon),
        num_distractors=data_config.get("num_distractors"),
        use_full_list=bool(data_config.get("use_full_station_list", True)),
        seed=int(config.get("seed", 42)),
    )


def build_dataloader(
    manifest_path: Path,
    config: dict[str, Any],
    processor: Any,
    tokenizer: Any,
    candidate_builder: StationCandidateBuilder,
    limit: int | None,
    shuffle: bool,
) -> DataLoader:
    dataset = AudioManifestDataset(
        manifest_path=manifest_path,
        repo_root=PROJECT_ROOT,
        use_null_samples=bool(config["data"].get("use_null_samples", True)),
        null_oversample_factor=int(config["data"].get("null_oversample_factor", 1)) if shuffle else 1,
        limit=limit,
    )
    collator = Stage2AudioQwenCollator(
        whisper_processor=processor,
        qwen_tokenizer=tokenizer,
        candidate_builder=candidate_builder,
        with_hotwords=bool(config["data"].get("with_hotwords", True)),
        target_sample_rate=int(config["data"]["target_sample_rate"]),
        max_prompt_length=int(config["data"]["max_prompt_length"]),
        max_answer_length=int(config["data"]["max_answer_length"]),
    )
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config["data"].get("num_workers", 0)),
        collate_fn=collator,
    )


def build_model_and_tokenizers(
    config: dict[str, Any],
    resume_from: str | Path | None = None,
) -> tuple[Any, Any, AudioQwenForStage2]:
    whisper_path = resolve_path(config["paths"]["whisper_model"])
    qwen_path = resolve_path(config["paths"]["qwen_model"])
    processor = AutoProcessor.from_pretrained(str(whisper_path), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(qwen_path), local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    training = config["training"]
    requested_device = str(training.get("device", "cuda:0"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {requested_device}, but CUDA is not available.")
    device = torch.device(requested_device)
    dtype = torch.float16 if device.type == "cuda" and bool(training.get("fp16", True)) else torch.float32

    model_config = config["model"]
    model = AudioQwenForStage2(
        whisper_model_path=whisper_path,
        qwen_model_path=qwen_path,
        adapter_type=str(model_config["adapter_type"]),
        num_query_tokens=int(model_config["num_query_tokens"]),
        adapter_hidden_size=int(model_config["adapter_hidden_size"]),
        adapter_dropout=float(model_config["adapter_dropout"]),
        freeze_whisper=bool(model_config.get("freeze_whisper", True)),
        torch_dtype=dtype,
    )
    model.load_stage1_checkpoint(resolve_path(config["paths"]["stage1_checkpoint"]))

    if resume_from is not None:
        resume_dir = resolve_path(resume_from)
        model.load_adapter_projector(resume_dir / "adapter_projector.pt")
        model.qwen = load_lora_adapter(model.qwen, resume_dir / "lora_adapter", is_trainable=True)
    else:
        lora = config["lora"]
        model.lora_target_info = lora_target_report(model.qwen, list(lora.get("target_modules", [])))  # type: ignore[attr-defined]
        model.enable_lora(
            target_modules=list(lora.get("target_modules", [])),
            r=int(lora["r"]),
            alpha=int(lora["alpha"]),
            dropout=float(lora["dropout"]),
            bias=str(lora.get("bias", "none")),
        )

    if bool(training.get("gradient_checkpointing", True)):
        model.qwen.gradient_checkpointing_enable()
        if hasattr(model.qwen, "config"):
            model.qwen.config.use_cache = False

    model.to(device)
    return processor, tokenizer, model


def print_trainable_parameters(model: AudioQwenForStage2) -> None:
    total, trainable = count_parameters(model)
    print(f"parameters total={total:,} trainable={trainable:,} ratio={trainable / total:.6f}")
    report = trainable_parameter_report(model)
    for name, count in report[:40]:
        print(f"trainable {name}: {count:,}")
    if len(report) > 40:
        print(f"... {len(report) - 40} more trainable tensors omitted")


def evaluate(model: AudioQwenForStage2, dataloader: DataLoader, fp16: bool) -> float:
    model.eval()
    losses: list[float] = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in dataloader:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda" and fp16):
                outputs = model(
                    input_features=batch["input_features"],
                    prompt_input_ids=batch["prompt_input_ids"],
                    prompt_attention_mask=batch["prompt_attention_mask"],
                    answer_input_ids=batch["answer_input_ids"],
                    answer_attention_mask=batch["answer_attention_mask"],
                )
            losses.append(float(outputs["loss"].detach().cpu()))
    model.train()
    return float(np.mean(losses)) if losses else math.inf


def save_checkpoint(output_dir: Path, model: AudioQwenForStage2, config: dict[str, Any], step: int, valid_loss: float) -> None:
    best_dir = output_dir / "best"
    target_dir = best_dir if not best_dir.exists() else output_dir / f"best_step_{step:06d}"
    if target_dir.exists():
        suffix = 1
        while (output_dir / f"{target_dir.name}_{suffix}").exists():
            suffix += 1
        target_dir = output_dir / f"{target_dir.name}_{suffix}"
    target_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "adapter": model.adapter.state_dict(),
            "projector": model.projector.state_dict(),
            "step": step,
            "valid_loss": valid_loss,
        },
        target_dir / "adapter_projector.pt",
    )
    model.qwen.save_pretrained(str(target_dir / "lora_adapter"))
    with (target_dir / "training_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    prompt_info = {
        "with_hotwords": bool(config["data"].get("with_hotwords", True)),
        "use_full_station_list": bool(config["data"].get("use_full_station_list", True)),
        "num_distractors": config["data"].get("num_distractors"),
        "step": step,
        "valid_loss": valid_loss,
    }
    with (target_dir / "prompt_config.json").open("w", encoding="utf-8") as f:
        json.dump(prompt_info, f, ensure_ascii=False, indent=2)
    pointer = output_dir / "best_checkpoint_dir.txt"
    pointer.write_text(target_dir.name, encoding="utf-8")
    print(f"saved checkpoint: {target_dir}")


def train_stage2(
    config_path: str | Path,
    limit_train: int | None = None,
    limit_valid: int | None = None,
    resume_from: str | Path | None = None,
) -> None:
    config = load_config(config_path)
    set_seed(int(config.get("seed", 42)))

    output_dir = resolve_path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"

    processor, tokenizer, model = build_model_and_tokenizers(config, resume_from=resume_from)
    print(
        "lora targets:",
        json.dumps(
            getattr(model, "lora_target_info", {"requested": config["lora"]["target_modules"], "available": "loaded_from_checkpoint"}),
            ensure_ascii=False,
        ),
    )
    print_trainable_parameters(model)

    candidate_builder = build_candidate_builder(config)
    train_loader = build_dataloader(
        resolve_manifest_path(config["paths"]["train_manifest"]),
        config,
        processor,
        tokenizer,
        candidate_builder,
        limit=limit_train,
        shuffle=True,
    )
    valid_loader = build_dataloader(
        resolve_manifest_path(config["paths"]["valid_manifest"]),
        config,
        processor,
        tokenizer,
        candidate_builder,
        limit=limit_valid,
        shuffle=False,
    )

    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = int(config["training"]["epochs"])
    grad_accum = int(config["training"]["gradient_accumulation_steps"])
    total_update_steps = max(1, math.ceil(len(train_loader) * epochs / grad_accum))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config["training"]["warmup_steps"]),
        num_training_steps=total_update_steps,
    )

    fp16 = bool(config["training"].get("fp16", True))
    log_steps = int(config["training"]["log_steps"])
    eval_steps = int(config["training"]["eval_steps"])
    max_grad_norm = float(config["training"]["max_grad_norm"])
    device = next(model.parameters()).device
    model.train()
    optimizer.zero_grad(set_to_none=True)

    best_valid = math.inf
    global_step = 0
    running_loss = 0.0
    start_time = time.time()

    with log_path.open("w", encoding="utf-8") as log_f:
        for epoch in range(epochs):
            for batch_index, batch in enumerate(tqdm(train_loader, desc=f"stage2 epoch {epoch + 1}"), start=1):
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda" and fp16):
                    outputs = model(
                        input_features=batch["input_features"],
                        prompt_input_ids=batch["prompt_input_ids"],
                        prompt_attention_mask=batch["prompt_attention_mask"],
                        answer_input_ids=batch["answer_input_ids"],
                        answer_attention_mask=batch["answer_attention_mask"],
                    )
                    loss = outputs["loss"] / grad_accum
                loss.backward()
                running_loss += float(loss.detach().cpu()) * grad_accum

                if batch_index % grad_accum != 0 and batch_index != len(train_loader):
                    continue

                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % log_steps == 0:
                    memory_mb = (
                        torch.cuda.max_memory_allocated(device) / (1024**2)
                        if device.type == "cuda"
                        else 0.0
                    )
                    row = {
                        "type": "train",
                        "step": global_step,
                        "epoch": epoch + 1,
                        "loss": running_loss / max(1, log_steps * grad_accum),
                        "lr": scheduler.get_last_lr()[0],
                        "cuda_memory_mb": round(memory_mb, 2),
                        "elapsed_sec": round(time.time() - start_time, 2),
                    }
                    print(json.dumps(row, ensure_ascii=False))
                    log_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    log_f.flush()
                    running_loss = 0.0

                if global_step % eval_steps == 0:
                    valid_loss = evaluate(model, valid_loader, fp16=fp16)
                    row = {"type": "eval", "step": global_step, "valid_loss": valid_loss}
                    print(json.dumps(row, ensure_ascii=False))
                    log_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    log_f.flush()
                    if valid_loss < best_valid:
                        best_valid = valid_loss
                        save_checkpoint(output_dir, model, config, global_step, valid_loss)

        final_valid = evaluate(model, valid_loader, fp16=fp16)
        row = {"type": "eval_final", "step": global_step, "valid_loss": final_valid}
        print(json.dumps(row, ensure_ascii=False))
        log_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if final_valid < best_valid:
            save_checkpoint(output_dir, model, config, global_step, final_valid)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_valid", type=int, default=None)
    parser.add_argument("--resume_from", default=None)
    args = parser.parse_args(argv)
    train_stage2(
        args.config,
        limit_train=args.limit_train,
        limit_valid=args.limit_valid,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
