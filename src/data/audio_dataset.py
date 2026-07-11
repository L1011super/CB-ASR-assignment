"""Manifest dataset for audio-to-station training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AudioSample:
    utt_id: str
    audio_path: Path
    text: str | None
    target_station: str | None
    has_station: bool
    raw: dict[str, Any]


class AudioManifestDataset:
    """Load JSONL manifest rows and keep only samples usable for stage 1."""

    def __init__(
        self,
        manifest_path: str | Path,
        repo_root: str | Path,
        use_null_samples: bool = False,
        null_oversample_factor: int = 1,
        limit: int | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.repo_root = Path(repo_root)
        self.use_null_samples = use_null_samples
        self.null_oversample_factor = max(1, int(null_oversample_factor))
        self.samples = self._load(limit)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> AudioSample:
        return self.samples[index]

    def _load(self, limit: int | None) -> list[AudioSample]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        samples: list[AudioSample] = []
        with self.manifest_path.open("r", encoding="utf-8-sig") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {self.manifest_path}:{line_no}: {exc}") from exc

                target_station = _normalize_station_value(row.get("target_station"))
                has_station = bool(row.get("has_station", target_station is not None))
                if target_station is None and not self.use_null_samples:
                    continue

                audio_value = row.get("processed_audio_path") or row.get("audio_path")
                if not audio_value:
                    continue
                audio_path = Path(str(audio_value))
                if not audio_path.is_absolute():
                    audio_path = self.repo_root / audio_path

                sample = AudioSample(
                    utt_id=str(row.get("utt_id") or f"row_{line_no}"),
                    audio_path=audio_path,
                    text=row.get("text"),
                    target_station=target_station,
                    has_station=has_station,
                    raw=row,
                )
                repeat = self.null_oversample_factor if target_station is None else 1
                samples.extend(sample for _ in range(repeat))
                if limit is not None and len(samples) >= limit:
                    samples = samples[:limit]
                    break

        if not samples:
            raise ValueError(f"No usable samples loaded from {self.manifest_path}")
        return samples


def _normalize_station_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "null" or text.upper() == "NULL":
        return None
    return text
