# Baselines

## Pure Whisper Raw ASR

`run_whisper_raw.py` runs a reproducible plain Whisper baseline for the Tianjin metro fixed-template station recognition task. It only sends each wav file to Whisper and writes the raw transcription as `asr_text`.

This baseline does not use station lexicon matching, pinyin correction, Qwen, contextual biasing, Adapter, or Q-Former logic.

## Input Manifest

The input is a JSONL file. Each line should contain an `utt_id` and either `processed_audio_path` or `audio_path`. If both audio fields exist, `processed_audio_path` is used first.

Common fields are preserved in the output when present:

```json
{
  "utt_id": "TJ014_tts_test_001",
  "audio_path": "data/processed/audio/TJ014_tts_test_001.wav",
  "processed_audio_path": "data/processed/audio/TJ014_tts_test_001.wav",
  "text": "我要去渌水道站",
  "target_station": "渌水道",
  "target_station_id": "TJ014",
  "has_station": true,
  "source": "tts",
  "split": "test",
  "voice": "zh-CN-XiaoxiaoNeural"
}
```

## Run

Install dependencies if the current conda environment is missing them:

```powershell
pip install transformers librosa soundfile tqdm pyyaml accelerate
```

Run with config:

```powershell
python baselines/run_whisper_raw.py --config configs/whisper_raw.yaml
```

Run on the current Kokoro TTS test manifest:

```powershell
python baselines/run_whisper_raw.py `
  --config configs/whisper_raw.yaml `
  --manifest data/manifests/test_tts.jsonl `
  --model_path pretrained/whisper-small `
  --output outputs/baseline/whisper_raw_test_predictions.jsonl `
  --error_output outputs/baseline/whisper_raw_errors.jsonl `
  --limit 20
```

The model must already exist locally. The script loads with `local_files_only=True` and will not download online weights.

## Output JSONL

Default output:

```text
outputs/baseline/whisper_raw_test_predictions.jsonl
```

Each successful row contains:

```json
{
  "utt_id": "...",
  "audio_path": "...",
  "target_text": "我要去渌水道站",
  "target_station": "渌水道",
  "target_station_id": "TJ014",
  "has_station": true,
  "source": "tts",
  "split": "test",
  "voice": "...",
  "asr_text": "我要去绿水道站",
  "method": "whisper_raw",
  "model_path": "pretrained/whisper-small",
  "duration_sec": 3.02,
  "decode_time_sec": 0.18
}
```

Errors are written to:

```text
outputs/baseline/whisper_raw_errors.jsonl
```

Use `--resume` to skip `utt_id` values already present in the prediction file. Use `--limit N` for a quick smoke test.

## Station Baselines 0-4

`run_whisper_station_baselines.py` runs Whisper once per audio sample and writes five baseline outputs:

- `baseline0_whisper_raw.jsonl`: audio -> Whisper -> raw `asr_text`
- `baseline1_exact_match.jsonl`: raw ASR text + exact canonical station-name match
- `baseline2_edit_fuzzy.jsonl`: fixed-template slot extraction + character edit-distance fuzzy match, with `top3`
- `baseline3_pinyin_fuzzy.jsonl`: fixed-template slot extraction + compact-pinyin fuzzy match, with `top3`
- `baseline4_lexicon_normalization.jsonl`: exact, alias/confusion, edit-distance, and pinyin normalization

Run on the current Kokoro TTS test split with GPU `cuda:0`:

```powershell
python baselines/run_whisper_station_baselines.py `
  --config configs/whisper_station_baselines.yaml `
  --no-resume
```

Default output directory:

```text
outputs/baseline/whisper_station_baselines/
```

The summary metrics are written to:

```text
outputs/baseline/whisper_station_baselines/summary.json
```

Use a small smoke test before the full run:

```powershell
python baselines/run_whisper_station_baselines.py `
  --config configs/whisper_station_baselines.yaml `
  --output_dir outputs/baseline/whisper_station_baselines_smoke `
  --error_output outputs/baseline/whisper_station_baselines_smoke/errors.jsonl `
  --limit 20 `
  --no-resume
```
