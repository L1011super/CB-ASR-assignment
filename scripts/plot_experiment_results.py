from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class MethodSpec:
    name: str
    pred_path: Path | None
    note: str


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_null_value(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in {"", "null", "none"}


def station_id_from_row(row: dict[str, Any]) -> str | None:
    station_id = row.get("target_station_id") or row.get("station_id")
    if station_id and str(station_id).upper() != "NULL":
        return str(station_id)
    utt_id = str(row.get("utt_id", ""))
    if utt_id.startswith("TJ") and "_" in utt_id:
        return utt_id.split("_", 1)[0]
    return None


def load_station_meta(station_csv: Path) -> dict[str, dict[str, bool]]:
    meta: dict[str, dict[str, bool]] = {}
    with station_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            station_id = str(row.get("station_id", "")).strip()
            if not station_id:
                continue
            meta[station_id] = {
                "is_rare": str(row.get("is_rare", "0")).strip() == "1",
                "has_confusions": bool(str(row.get("confusions", "")).strip()),
            }
    return meta


def compute_metrics(
    rows: list[dict[str, Any]],
    station_meta: dict[str, dict[str, bool]],
    has_station_prediction: bool = True,
) -> dict[str, Any]:
    groups = {
        "overall": {"total": 0, "correct": 0},
        "station": {"total": 0, "correct": 0},
        "null": {"total": 0, "correct": 0},
        "rare": {"total": 0, "correct": 0},
        "confusion": {"total": 0, "correct": 0},
    }
    decode_times: list[float] = []

    for row in rows:
        if isinstance(row.get("decode_time_sec"), int | float):
            decode_times.append(float(row["decode_time_sec"]))

        target = row.get("target_station")
        pred = row.get("pred_station")
        target_is_null = is_null_value(target)

        if has_station_prediction:
            correct = (target_is_null and is_null_value(pred)) or (
                not target_is_null and str(pred) == str(target)
            )
        else:
            correct = False

        groups["overall"]["total"] += 1
        groups["overall"]["correct"] += int(correct)

        if target_is_null:
            groups["null"]["total"] += 1
            groups["null"]["correct"] += int(correct)
            continue

        groups["station"]["total"] += 1
        groups["station"]["correct"] += int(correct)

        station_id = station_id_from_row(row)
        station_info = station_meta.get(station_id or "", {})
        if station_info.get("is_rare", False):
            groups["rare"]["total"] += 1
            groups["rare"]["correct"] += int(correct)
        if station_info.get("has_confusions", False):
            groups["confusion"]["total"] += 1
            groups["confusion"]["correct"] += int(correct)

    result: dict[str, Any] = {}
    for key, value in groups.items():
        total = value["total"]
        correct = value["correct"]
        result[f"{key}_total"] = total
        result[f"{key}_correct"] = correct
        result[f"{key}_accuracy"] = (
            correct / total if total and has_station_prediction else None
        )
    result["avg_decode_time_sec"] = (
        sum(decode_times) / len(decode_times) if decode_times else None
    )
    result["total_errors"] = (
        result["overall_total"] - result["overall_correct"]
        if has_station_prediction
        else None
    )
    return result


def pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def numeric_or_na(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_metrics_tables(metrics: dict[str, dict[str, Any]], output_dir: Path) -> None:
    rows = []
    for method, item in metrics.items():
        rows.append(
            {
                "method": method,
                "overall_acc": pct(item.get("overall_accuracy")),
                "station_acc": pct(item.get("station_accuracy")),
                "null_acc": pct(item.get("null_accuracy")),
                "rare_acc": pct(item.get("rare_accuracy")),
                "confusion_acc": pct(item.get("confusion_accuracy")),
                "correct/total": (
                    f"{item.get('overall_correct', 'N/A')}/{item.get('overall_total', 'N/A')}"
                    if item.get("overall_accuracy") is not None
                    else "N/A"
                ),
                "errors": numeric_or_na(item.get("total_errors"), 0),
                "avg_decode_time_sec": numeric_or_na(item.get("avg_decode_time_sec"), 4),
                "note": str(item.get("note", "")),
            }
        )

    csv_path = output_dir / "experiment_metrics_table.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_dir / "experiment_metrics_table.md"
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_accuracy_bar(metrics: dict[str, dict[str, Any]], output_dir: Path) -> None:
    plot_items = [
        (method, item["overall_accuracy"])
        for method, item in metrics.items()
        if item.get("overall_accuracy") is not None
    ]
    labels = [x[0] for x in plot_items]
    values = [float(x[1]) * 100 for x in plot_items]

    fig_width = max(10, len(labels) * 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    colors = ["#8492a6", "#6b8fbf", "#5aa469", "#c9a227", "#d95f59"]
    bars = ax.bar(labels, values, color=colors[: len(labels)], edgecolor="#333333")

    ax.set_ylim(0, 105)
    ax.set_ylabel("Overall Accuracy (%)")
    ax.set_title("Station Recognition Accuracy on Test Set")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.bar_label(bars, labels=[f"{v:.2f}%" for v in values], padding=3, fontsize=10)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()

    png_path = output_dir / "accuracy_bar_overall.png"
    pdf_path = output_dir / "accuracy_bar_overall.pdf"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--station_csv", default="data/stations/tianjin_metro_stations.csv")
    parser.add_argument(
        "--baseline_dir", default="outputs/baseline/whisper_station_baselines"
    )
    parser.add_argument(
        "--ours_pred",
        default="outputs/preds/stage2_hotwords_attention_q32_hard_refine_test_best.jsonl",
    )
    parser.add_argument("--output_dir", default="outputs/report_figures")
    args = parser.parse_args()

    station_meta = load_station_meta(Path(args.station_csv))
    baseline_dir = Path(args.baseline_dir)
    specs = [
        MethodSpec(
            "B1 Exact Match",
            baseline_dir / "baseline1_exact_match.jsonl",
            "Whisper ASR + exact station-name substring match.",
        ),
        MethodSpec(
            "B2 Edit Fuzzy",
            baseline_dir / "baseline2_edit_fuzzy.jsonl",
            "Whisper ASR + slot extraction + edit-distance fuzzy match.",
        ),
        MethodSpec(
            "B3 Pinyin Fuzzy",
            baseline_dir / "baseline3_pinyin_fuzzy.jsonl",
            "Whisper ASR + slot extraction + pinyin similarity.",
        ),
        MethodSpec(
            "B4 Lexicon Norm",
            baseline_dir / "baseline4_lexicon_normalization.jsonl",
            "Whisper ASR + station lexicon normalization.",
        ),
        MethodSpec(
            "Ours Optimized",
            Path(args.ours_pred),
            "Whisper Encoder + Adapter + Qwen + hotword list + LoRA + hard-refine.",
        ),
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: dict[str, dict[str, Any]] = {}
    for spec in specs:
        if spec.pred_path is None or not spec.pred_path.exists():
            raise FileNotFoundError(f"Missing prediction file: {spec.pred_path}")
        rows = read_jsonl(spec.pred_path)
        item = compute_metrics(rows, station_meta, has_station_prediction=True)
        item["note"] = spec.note
        all_metrics[spec.name] = item

    # Baseline 0 is ASR text only. It is not plotted because it does not produce
    # a station prediction, but its decode statistic is useful for the report.
    raw_path = baseline_dir / "baseline0_whisper_raw.jsonl"
    if raw_path.exists():
        raw_rows = read_jsonl(raw_path)
        raw_item = compute_metrics(raw_rows, station_meta, has_station_prediction=False)
        raw_item["note"] = "Whisper raw transcription only; no station prediction."
        all_metrics = {"B0 Whisper Raw": raw_item, **all_metrics}

    plot_metrics = {
        key: value
        for key, value in all_metrics.items()
        if key != "B0 Whisper Raw"
    }
    plot_accuracy_bar(plot_metrics, output_dir)
    write_metrics_tables(all_metrics, output_dir)

    summary_path = output_dir / "experiment_metrics_summary.json"
    summary_path.write_text(
        json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Wrote:")
    print(f"  {output_dir / 'accuracy_bar_overall.png'}")
    print(f"  {output_dir / 'accuracy_bar_overall.pdf'}")
    print(f"  {output_dir / 'experiment_metrics_table.csv'}")
    print(f"  {output_dir / 'experiment_metrics_table.md'}")
    print(f"  {summary_path}")
    print()
    print((output_dir / "experiment_metrics_table.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
