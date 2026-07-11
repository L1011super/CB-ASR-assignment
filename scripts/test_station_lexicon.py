"""Smoke tests for station lexicon loading and normalization.

Run from the project root:
    python scripts/test_station_lexicon.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.station_lexicon import (  # noqa: E402
    build_hotword_prompt,
    load_station_lexicon,
    normalize_station_name,
    station_names,
)


YINGKOUDAO = "\u8425\u53e3\u9053"
YINGKOUDAO_WRONG_1 = "\u8425\u53e3\u5230"
YINGKOUDAO_WRONG_2 = "\u8fce\u53e3\u9053"
LUSHUIDAO = "\u6e0c\u6c34\u9053"
LISHUIDAO = "\u6fa7\u6c34\u9053"
def main() -> None:
    lexicon_path = PROJECT_ROOT / "data" / "stations" / "tianjin_metro_stations.csv"
    lexicon = load_station_lexicon(lexicon_path)

    names = station_names(lexicon)
    assert YINGKOUDAO in names
    assert LUSHUIDAO in names
    assert LISHUIDAO in names

    assert normalize_station_name(YINGKOUDAO_WRONG_1, lexicon) == YINGKOUDAO
    assert normalize_station_name(YINGKOUDAO_WRONG_2, lexicon) == YINGKOUDAO

    # These are both real station names, so normalization must preserve them.
    assert normalize_station_name(LUSHUIDAO, lexicon) == LUSHUIDAO
    assert normalize_station_name(LISHUIDAO, lexicon) == LISHUIDAO

    prompt = build_hotword_prompt(lexicon, max_items=2)
    assert names[0] in prompt
    assert names[1] in prompt
    assert names[2] not in prompt

    print("station_lexicon smoke tests passed")


if __name__ == "__main__":
    main()
