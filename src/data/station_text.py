"""Text templates for Tianjin metro fixed-phrase TTS data."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rapidfuzz import fuzz

from src.data.station_lexicon import StationEntry, load_station_lexicon


@dataclass(frozen=True)
class StationTextSample:
    station_id: str
    target_station: str | None
    text: str
    has_station: bool
    station_pinyin: str | None = None


THINKING_FILLERS: tuple[str, ...] = ("\u55ef", "\u90a3\u4e2a", "\u5443")


ILLEGAL_DESTINATIONS: tuple[str, ...] = (
    "水上公园",
    "南开大学",
    "天津大学",
    "北京南",
    "商场",
    "医院",
    "图书馆",
    "机场",
    "学校",
    "电影院",
    "博物馆",
    "饭店",
    "体育馆",
    "海河边",
    "天津之眼",
    "古文化街",
    "北京站",
    "上海站",
    "便利店",
    "药店",
    "健身房",
    "咖啡馆",
    "南开大学津南校区",
    "天津大学北洋园校区",
    "八里台校区",
    "海教园",
    "梅江会展中心",
    "奥城商业广场",
    "滨海机场",
    "天津西青大学城",
    "和平路商圈",
    "五大道景区",
    "意式风情区",
    "瓷房子",
    "鼓楼商业街",
    "大悦城",
    "恒隆广场",
    "万象城",
    "银河购物中心",
    "永旺梦乐城",
    "宜家家居",
    "欢乐谷",
    "方特欢乐世界",
    "动物园",
    "植物园",
    "人民公园",
    "中心公园",
    "大学城",
    "菜市场",
    "水果店",
    "理发店",
    "银行",
    "派出所",
    "社区医院",
    "儿童医院",
    "总医院",
    "肿瘤医院",
    "眼科医院",
    "口腔医院",
    "市政府",
    "区政府",
    "税务局",
    "法院",
    "邮局",
    "快递站",
    "公交站",
    "长途客运站",
    "火车票售票处",
    "北京西",
    "北京北",
    "上海虹桥",
    "广州南",
    "深圳北",
    "济南西",
    "石家庄站",
    "唐山站",
    "秦皇岛站",
    "沧州西",
    "廊坊站",
    "保定东",
    "太原南",
    "郑州东",
    "南京南",
    "杭州东",
    "成都东",
    "西安北",
    "武汉站",
    "长沙南",
    "沈阳北",
    "哈尔滨西",
    "超市",
    "诊所",
    "酒店",
    "宾馆",
    "食堂",
    "实验楼",
    "教学楼",
    "宿舍楼",
    "操场",
    "游泳馆",
    "音乐厅",
    "美术馆",
    "科技馆",
    "档案馆",
    "会议中心",
)


def load_stations(path: str | Path) -> list[StationEntry]:
    """Load canonical station rows from the project CSV."""

    return load_station_lexicon(path)


def station_destination_text(station_name: str) -> str:
    """Return the station phrase spoken after the fixed prefix."""

    if station_name.endswith("\u7ad9"):
        return station_name
    return f"{station_name}\u7ad9"


def positive_text(station_name: str, filler: str = "") -> str:
    """Return the fixed positive command for one canonical station name."""

    return f"{filler}\u6211\u8981\u53bb{station_destination_text(station_name)}"


def sample_thinking_filler(
    rng: random.Random,
    probability: float,
    fillers: Iterable[str] = THINKING_FILLERS,
) -> str:
    """Return a short hesitation filler with a low fixed probability."""

    if probability <= 0 or rng.random() >= probability:
        return ""
    pool = tuple(item for item in fillers if item)
    return rng.choice(pool) if pool else ""


def null_text(destination: str) -> str:
    """Return a fixed-phrase null command."""

    dest = destination[:-1] if destination.endswith("\u7ad9") else destination
    return f"\u6211\u8981\u53bb{dest}\u7ad9"


def build_positive_samples(stations: Iterable[StationEntry]) -> list[StationTextSample]:
    return [
        StationTextSample(
            station_id=station.station_id,
            target_station=station.name,
            text=positive_text(station.name),
            has_station=True,
            station_pinyin=station.pinyin,
        )
        for station in stations
    ]


def build_null_destinations(station_names: Iterable[str]) -> list[str]:
    """Filter built-in null destinations against legal station names."""

    legal_names = set(station_names)
    filtered: list[str] = []
    for candidate in ILLEGAL_DESTINATIONS:
        normalized = candidate[:-1] if candidate.endswith("\u7ad9") else candidate
        if normalized in legal_names or candidate in legal_names:
            continue
        if _too_close_to_station(normalized, legal_names):
            continue
        if normalized not in filtered:
            filtered.append(normalized)
    if len(filtered) < 80:
        raise ValueError(f"Need at least 80 illegal destinations, got {len(filtered)}")
    return filtered


def sample_null_destinations(
    count: int,
    rng: random.Random,
    station_names: Iterable[str],
) -> list[str]:
    """Sample illegal destinations with replacement for fixed null commands."""

    pool = build_null_destinations(station_names)
    return [rng.choice(pool) for _ in range(count)]


def _too_close_to_station(candidate: str, station_names: Iterable[str]) -> bool:
    for station in station_names:
        short_station = station[:-1] if station.endswith("\u7ad9") else station
        if fuzz.ratio(candidate, short_station) >= 92:
            return True
    return False
