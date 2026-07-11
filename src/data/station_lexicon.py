"""Station lexicon loading and station-name normalization utilities.

The module keeps station metadata small and explicit. CSV rows are validated on
load, while pinyin fields are generated from the canonical station name so later
ASR post-processing does not depend on hand-written pinyin being correct.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    from pypinyin import Style, lazy_pinyin
except ImportError as exc:  # pragma: no cover - exercised only without deps.
    raise ImportError(
        "station_lexicon requires pypinyin. Install it with `pip install pypinyin`."
    ) from exc

try:
    from rapidfuzz import fuzz, process
except ImportError as exc:  # pragma: no cover - exercised only without deps.
    raise ImportError(
        "station_lexicon requires rapidfuzz. Install it with `pip install rapidfuzz`."
    ) from exc


REQUIRED_COLUMNS = {
    "station_id",
    "name",
    "line_ids",
    "pinyin",
    "pinyin_compact",
    "is_rare",
}
SPLIT_PATTERN = re.compile(r"[|;\uff1b]")

# Conservative thresholds: exact aliases/confusions can be trusted, but fuzzy
# matches must be strong enough to avoid merging real near-homophone stations.
EDIT_DISTANCE_THRESHOLD = 92.0
PINYIN_DISTANCE_THRESHOLD = 96.0
AMBIGUITY_MARGIN = 3.0


@dataclass(frozen=True)
class StationEntry:
    """One canonical station and its matching metadata."""

    station_id: str
    name: str
    line_ids: tuple[str, ...]
    pinyin: str
    pinyin_compact: str
    aliases: tuple[str, ...]
    confusions: tuple[str, ...]
    is_rare: bool


StationLexicon = list[StationEntry]


def load_station_lexicon(path: str | Path) -> StationLexicon:
    """Load station metadata from CSV and return canonical station entries.

    Empty placeholder files are accepted and return an empty list. For real CSV
    files, all required columns must exist. Pinyin values are generated from
    ``name``; existing ``pinyin`` and ``pinyin_compact`` columns are validated.
    """

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Station lexicon CSV not found: {csv_path}")

    if csv_path.stat().st_size == 0:
        return []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fieldnames
        if missing:
            missing_cols = ", ".join(sorted(missing))
            raise ValueError(f"Station lexicon missing required columns: {missing_cols}")

        entries: StationLexicon = []
        seen_ids: set[str] = set()
        seen_names: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            entry = _parse_station_row(row, row_number)
            if entry.station_id in seen_ids:
                raise ValueError(f"Duplicate station_id at row {row_number}: {entry.station_id}")
            if entry.name in seen_names:
                raise ValueError(f"Duplicate station name at row {row_number}: {entry.name}")
            seen_ids.add(entry.station_id)
            seen_names.add(entry.name)
            entries.append(entry)

    return entries


def normalize_station_name(text: str, lexicon: Sequence[StationEntry]) -> str | None:
    """Normalize ASR text to a canonical station name.

    Matching order:
    1. exact canonical station-name match
    2. exact alias/confusion match
    3. rapidfuzz edit-distance match
    4. pypinyin compact-pinyin similarity match

    Ambiguous fuzzy matches are rejected by returning ``None``.
    """

    query = _clean_station_text(text)
    if not query or not lexicon:
        return None

    exact_name = {entry.name: entry.name for entry in lexicon}
    if query in exact_name:
        return exact_name[query]

    alias_map = _build_alias_map(lexicon)
    if query in alias_map:
        return alias_map[query]

    edit_match = _best_unique_match(
        query=query,
        choices=[entry.name for entry in lexicon],
        threshold=EDIT_DISTANCE_THRESHOLD,
    )
    if edit_match is not None:
        return edit_match

    query_pinyin = _pinyin_compact(query)
    pinyin_to_name = {entry.pinyin_compact: entry.name for entry in lexicon}
    pinyin_match = _best_unique_match(
        query=query_pinyin,
        choices=list(pinyin_to_name),
        threshold=PINYIN_DISTANCE_THRESHOLD,
    )
    if pinyin_match is None:
        return None
    return pinyin_to_name[pinyin_match]


def build_hotword_prompt(lexicon: Sequence[StationEntry], max_items: int | None = None) -> str:
    """Build a compact station-name prompt for contextual biasing."""

    entries = list(lexicon[:max_items] if max_items is not None else lexicon)
    names = "\u3001".join(entry.name for entry in entries)
    if not names:
        return "\u5929\u6d25\u5730\u94c1\u5019\u9009\u7ad9\u540d\u8868\u4e3a\u7a7a\u3002"
    return (
        "\u5929\u6d25\u5730\u94c1\u5019\u9009\u7ad9\u540d\u5982\u4e0b\uff0c"
        f"\u53ea\u80fd\u8f93\u51fa\u5176\u4e2d\u4e00\u4e2a\u6807\u51c6\u7ad9\u540d\u6216 null\uff1a{names}"
    )


def station_names(lexicon: Iterable[StationEntry]) -> list[str]:
    """Return only canonical station names from a loaded lexicon."""

    return [entry.name for entry in lexicon]


def _parse_station_row(row: dict[str, str], row_number: int) -> StationEntry:
    station_id = _required_value(row, "station_id", row_number)
    name = _required_value(row, "name", row_number)
    line_ids = tuple(_split_multi_value(_required_value(row, "line_ids", row_number)))
    is_rare_text = _required_value(row, "is_rare", row_number)

    if is_rare_text not in {"0", "1"}:
        raise ValueError(f"Row {row_number} has invalid is_rare value: {is_rare_text}")

    csv_pinyin = _required_value(row, "pinyin", row_number)
    csv_compact = _required_value(row, "pinyin_compact", row_number)
    normalized_pinyin = _normalize_pinyin(csv_pinyin)
    normalized_compact = _normalize_pinyin(csv_compact).replace(" ", "")
    if normalized_compact != normalized_pinyin.replace(" ", ""):
        raise ValueError(
            f"Row {row_number} pinyin_compact mismatch for {name}: "
            f"expected `{normalized_pinyin.replace(' ', '')}`, got `{csv_compact}`"
        )

    return StationEntry(
        station_id=station_id,
        name=name,
        line_ids=line_ids,
        pinyin=normalized_pinyin,
        pinyin_compact=normalized_compact,
        aliases=tuple(_split_multi_value(row.get("aliases", ""))),
        confusions=tuple(_split_multi_value(row.get("confusions", ""))),
        is_rare=is_rare_text == "1",
    )


def _required_value(row: dict[str, str], column: str, row_number: int) -> str:
    value = (row.get(column) or "").strip()
    if not value:
        raise ValueError(f"Row {row_number} missing required value: {column}")
    return value


def _split_multi_value(value: str) -> list[str]:
    return [item.strip() for item in SPLIT_PATTERN.split(value or "") if item.strip()]


def _build_alias_map(lexicon: Sequence[StationEntry]) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    ambiguous: set[str] = set()
    for entry in lexicon:
        for alias in (*entry.aliases, *entry.confusions):
            cleaned = _clean_station_text(alias)
            if not cleaned:
                continue
            previous = alias_map.get(cleaned)
            if previous is not None and previous != entry.name:
                ambiguous.add(cleaned)
            alias_map[cleaned] = entry.name
    for alias in ambiguous:
        alias_map.pop(alias, None)
    return alias_map


def _best_unique_match(query: str, choices: Sequence[str], threshold: float) -> str | None:
    if not choices:
        return None

    matches = process.extract(query, choices, scorer=fuzz.ratio, limit=2)
    if not matches:
        return None

    best_choice, best_score, _ = matches[0]
    if best_score < threshold:
        return None
    if len(matches) > 1:
        _, second_score, _ = matches[1]
        if best_score - second_score < AMBIGUITY_MARGIN:
            return None
    return str(best_choice)


def _clean_station_text(text: str) -> str:
    cleaned = re.sub(r"\s+", "", text or "")
    for suffix in ("\u5730\u94c1\u7ad9", "\u7ad9"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
    return cleaned


def _pinyin_with_spaces(text: str) -> str:
    syllables = lazy_pinyin(text, style=Style.NORMAL, errors="ignore", strict=False)
    return " ".join(syllable.lower() for syllable in syllables if syllable)


def _pinyin_compact(text: str) -> str:
    return _pinyin_with_spaces(text).replace(" ", "")


def _normalize_pinyin(text: str) -> str:
    return " ".join((text or "").lower().split())
