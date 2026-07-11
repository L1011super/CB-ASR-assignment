"""Prompt templates for audio-conditioned station recognition."""

from __future__ import annotations

import json
import random
from collections.abc import Sequence


STAGE_SYSTEM_PROMPT = "你是天津地铁自助售票机的语音站名识别模块。"

STAGE1_PROMPT = (
    "请根据用户语音识别目的地站名。"
    "请严格输出 JSON，不要解释。"
    '输出格式：{"station": "标准站名"}。'
)


def build_stage1_prompt() -> str:
    """Return the fixed stage-1 user prompt without hotword candidates."""

    return STAGE1_PROMPT


def build_stage1_answer(station: str | None) -> str:
    """Return the target JSON string for one sample."""

    return json.dumps({"station": station}, ensure_ascii=False)


def build_stage2_prompt(candidates: Sequence[str] | None = None, with_hotwords: bool = True) -> str:
    """Return the stage-2 user instruction, optionally with station hotwords."""

    parts = [
        "请根据用户语音识别目的地站名。",
        "请严格输出 JSON，不要解释。",
    ]
    if with_hotwords:
        candidate_text = "、".join(candidates or [])
        parts.extend(
            [
                f"候选天津地铁标准站名如下：{candidate_text}。",
                "你只能从候选站名中选择一个完整的标准站名；如果语音中的目的地不是候选站名，必须输出 null。",
                "不要把非地铁地点、学校、商场、医院或闲聊内容强行归一化为候选站名。",
            ]
        )
    else:
        parts.append("如果语音中没有合法天津地铁站名，必须输出 null。")
    parts.append('输出格式：{"station": "标准站名"} 或 {"station": null}。')
    return "".join(parts)


def build_stage2_answer(station: str | None) -> str:
    """Return the target JSON string for stage-2 training/inference."""

    return json.dumps({"station": station}, ensure_ascii=False)


class StationCandidateBuilder:
    """Build per-sample station candidate lists for contextual biasing."""

    def __init__(
        self,
        station_names: Sequence[str],
        num_distractors: int | None = None,
        use_full_list: bool = True,
        seed: int = 42,
    ) -> None:
        self.station_names = list(dict.fromkeys(station_names))
        if not self.station_names:
            raise ValueError("Station candidate list is empty.")
        self.num_distractors = num_distractors
        self.use_full_list = use_full_list
        self.rng = random.Random(seed)

    def build(self, target_station: str | None) -> list[str]:
        """Return candidates; positive samples always include the target station."""

        if self.use_full_list or self.num_distractors is None:
            return list(self.station_names)

        pool = [name for name in self.station_names if name != target_station]
        count = min(max(0, int(self.num_distractors)), len(pool))
        candidates = self.rng.sample(pool, count) if count else []
        if target_station is not None:
            if target_station not in self.station_names:
                raise ValueError(f"Target station is not in station list: {target_station}")
            candidates.append(target_station)
        self.rng.shuffle(candidates)
        return candidates
