"""Run YOLO validation and export F1/precision/recall curve data.

Examples:
    python script/validate_f1_curves.py --model yolo26n.pt --data data-1.yaml
    python script/validate_f1_curves.py --model runs/train/exp/weights/best.pt --data coco8.yaml --device 0
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ULTRALYTICS_REPO = WORKSPACE_ROOT / "ultralytics-main"
LOCAL_ULTRALYTICS_PACKAGE = LOCAL_ULTRALYTICS_REPO / "ultralytics"
DEFAULT_PROJECT = WORKSPACE_ROOT / "runs" / "val_curves"
DEFAULT_NAME = "val"


def main() -> int:
    args = parse_args()
    ensure_local_ultralytics_repo()

    model_path = resolve_cli_path("model", args.model)
    data_path = resolve_cli_path("data", args.data)
    project = resolve_cli_path("project", args.project)

    val_args = compact_dict(
        {
            "data": data_path,
            "split": args.split,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "workers": args.workers,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "half": args.half,
            "save_json": args.save_json,
            "save_txt": args.save_txt,
            "save_conf": args.save_conf,
            "plots": True,
            "project": project,
            "name": args.name,
            "exist_ok": args.exist_ok,
            "verbose": args.verbose,
        }
    )

    print_run_header(model_path=model_path, data_path=data_path, val_args=val_args, task=args.task)
    model = build_model(model_path, args.task)
    metrics = model.val(**val_args)

    save_dir = Path(getattr(metrics, "save_dir", Path(project) / args.name)).resolve()
    export_dir = save_dir / "curve_data"

    exported = export_validation_data(
        metrics=metrics,
        export_dir=export_dir,
        model_path=model_path,
        data_path=data_path,
        val_args=val_args,
        save_dir=save_dir,
    )

    print("\nValidation finished.")
    print(f"Run directory : {save_dir}")
    print(f"Curve data    : {export_dir}")
    for label, path in exported.items():
        print(f"{label:<14}: {path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a YOLO model on the validation split and export F1/PR/P/R curve arrays.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", "--weights", required=True, dest="model", help="Model weights or model YAML path.")
    parser.add_argument("--data", required=True, help="Dataset YAML path or Ultralytics dataset reference.")
    parser.add_argument(
        "--task",
        choices=["detect", "segment", "pose", "obb"],
        default="detect",
        help="YOLO task type used when constructing the model.",
    )
    parser.add_argument("--split", choices=["val", "test", "train"], default="val", help="Dataset split to validate.")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
    parser.add_argument("--batch", type=int, default=8, help="Validation batch size.")
    parser.add_argument("--device", default=None, help="Device, for example 0, 0,1, cpu, or mps.")
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader workers.")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold. None lets Ultralytics choose.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=3000, help="Maximum detections per image.")
    parser.add_argument("--half", action="store_true", help="Use FP16 validation when supported.")
    parser.add_argument("--save-json", action="store_true", help="Save COCO-format prediction JSON when supported.")
    parser.add_argument("--save-txt", action="store_true", help="Save prediction labels as txt files.")
    parser.add_argument("--save-conf", action="store_true", help="Include confidence values with saved txt labels.")
    parser.add_argument("--project", default=str(DEFAULT_PROJECT), help="Output project directory.")
    parser.add_argument("--name", default=DEFAULT_NAME, help="Run name under the project directory.")
    parser.add_argument("--exist-ok", action="store_true", help="Reuse the run directory if it already exists.")
    parser.add_argument("--verbose", action="store_true", help="Print per-class validation metrics.")
    return parser.parse_args()


def export_validation_data(
    metrics: Any,
    export_dir: Path,
    model_path: str,
    data_path: str,
    val_args: dict[str, Any],
    save_dir: Path,
) -> dict[str, str]:
    export_dir.mkdir(parents=True, exist_ok=True)

    overall_metrics = to_plain_data(getattr(metrics, "results_dict", {}))
    per_class_metrics = get_per_class_metrics(metrics)
    curves = collect_curves(metrics)
    plot_files = find_plot_files(save_dir)

    summary = {
        "model": model_path,
        "data": data_path,
        "split": val_args.get("split"),
        "save_dir": str(save_dir),
        "export_dir": str(export_dir),
        "overall_metrics": overall_metrics,
        "per_class_metrics": per_class_metrics,
        "curve_names": [curve["name"] for curve in curves],
        "plot_files": [str(path) for path in plot_files],
        "speed": to_plain_data(getattr(metrics, "speed", {})),
    }

    paths: dict[str, Path] = {
        "summary_json": export_dir / "summary.json",
        "overall_csv": export_dir / "overall_metrics.csv",
        "per_class_csv": export_dir / "per_class_metrics.csv",
        "curves_json": export_dir / "curves.json",
    }

    write_json(paths["summary_json"], summary)
    write_json(paths["curves_json"], curves)
    write_key_value_csv(paths["overall_csv"], overall_metrics)
    write_table_csv(paths["per_class_csv"], per_class_metrics)

    for curve in curves:
        curve_path = export_dir / f"{safe_file_stem(curve['name'])}.csv"
        write_curve_csv(curve_path, curve)
        paths[f"curve_{safe_file_stem(curve['name'])}"] = curve_path

    return {label: str(path) for label, path in paths.items()}


def collect_curves(metrics: Any) -> list[dict[str, Any]]:
    curve_names = list(getattr(metrics, "curves", []) or [])
    curve_results = list(getattr(metrics, "curves_results", []) or [])
    class_indices = list(to_plain_data(getattr(metrics, "ap_class_index", [])) or [])
    names = normalize_names(getattr(metrics, "names", {}) or {})

    curves: list[dict[str, Any]] = []
    for index, values in enumerate(curve_results):
        if len(values) != 4:
            continue

        x_values, y_values, x_label, y_label = to_plain_data(values)
        rows = normalize_2d(y_values)
        series = []
        for row_index, y_row in enumerate(rows):
            class_index = class_indices[row_index] if row_index < len(class_indices) else row_index
            series.append(
                {
                    "class_index": class_index,
                    "class_name": names.get(class_index, str(class_index)),
                    "y": y_row,
                }
            )

        curves.append(
            {
                "name": curve_names[index] if index < len(curve_names) else f"curve_{index}",
                "x_label": x_label,
                "y_label": y_label,
                "x": list(x_values),
                "series": series,
            }
        )

    return curves


def get_per_class_metrics(metrics: Any) -> list[dict[str, Any]]:
    summary = getattr(metrics, "summary", None)
    if not callable(summary):
        return []

    try:
        return to_plain_data(summary(normalize=True, decimals=8))
    except Exception as exc:
        return [{"error": f"Could not export per-class metrics: {exc}"}]


def write_curve_csv(path: Path, curve: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["curve", "class_index", "class_name", "x_label", "y_label", "x", "y"],
        )
        writer.writeheader()
        for series in curve["series"]:
            for x_value, y_value in zip(curve["x"], series["y"]):
                writer.writerow(
                    {
                        "curve": curve["name"],
                        "class_index": series["class_index"],
                        "class_name": series["class_name"],
                        "x_label": curve["x_label"],
                        "y_label": curve["y_label"],
                        "x": x_value,
                        "y": y_value,
                    }
                )


def write_table_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_key_value_csv(path: Path, values: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in values.items():
            writer.writerow({"metric": key, "value": value})


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def find_plot_files(save_dir: Path) -> list[Path]:
    patterns = ("*F1_curve.png", "*P_curve.png", "*R_curve.png", "*PR_curve.png")
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(save_dir.glob(pattern)))
    return paths


def compact_dict(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def resolve_cli_path(key: str, value: str) -> str:
    return str(resolve_path_value(key=key, value=value))


def resolve_path_value(key: str, value: Any) -> Any:
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return value

    if key == "data" and text.startswith("ul://"):
        return value

    path = Path(text).expanduser()
    local_ultralytics_path = resolve_local_ultralytics_reference(key=key, path=path)
    if local_ultralytics_path is not None:
        return str(local_ultralytics_path)

    if path.is_absolute():
        return str(path)

    if key == "project":
        return str((WORKSPACE_ROOT / path).resolve())

    candidates = [WORKSPACE_ROOT / path]

    if key == "data":
        candidates.append(LOCAL_ULTRALYTICS_PACKAGE / "cfg" / "datasets" / path.name)
    elif key == "model" and path.suffix in {".yaml", ".yml"}:
        matches = list((LOCAL_ULTRALYTICS_PACKAGE / "cfg" / "models").rglob(path.name))
        if len(matches) == 1:
            candidates.append(matches[0])

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    return value


def resolve_local_ultralytics_reference(key: str, path: Path) -> Path | None:
    if key != "model":
        return None

    parts = path.parts
    candidate: Path | None = None
    if "ultralytics-main" in parts:
        index = parts.index("ultralytics-main")
        candidate = LOCAL_ULTRALYTICS_REPO.joinpath(*parts[index + 1 :])
    elif "ultralytics" in parts:
        index = parts.index("ultralytics")
        candidate = LOCAL_ULTRALYTICS_PACKAGE.joinpath(*parts[index + 1 :])

    if candidate is not None and candidate.exists():
        return candidate.resolve()
    return None


def build_model(model_value: str, task: str | None = None):
    ensure_local_ultralytics_repo()

    model_text = str(model_value)
    model_stem = Path(model_text).stem.lower()

    if "rtdetr" in model_stem:
        from ultralytics import RTDETR

        return RTDETR(model_text)
    if "fastsam" in model_stem:
        from ultralytics import FastSAM

        return FastSAM(model_text)
    if "sam_" in model_stem or "sam2_" in model_stem or "sam2.1_" in model_stem:
        from ultralytics import SAM

        return SAM(model_text)

    from ultralytics import YOLO

    return YOLO(model_text, task=task)


def ensure_local_ultralytics_repo() -> None:
    if not LOCAL_ULTRALYTICS_REPO.exists():
        raise FileNotFoundError(f"Local Ultralytics repo not found: {LOCAL_ULTRALYTICS_REPO}")

    repo_text = str(LOCAL_ULTRALYTICS_REPO.resolve())
    for existing in (repo_text, str(LOCAL_ULTRALYTICS_REPO)):
        while existing in sys.path:
            sys.path.remove(existing)
    sys.path.insert(0, repo_text)


def normalize_names(names: Any) -> dict[int, str]:
    if not isinstance(names, dict):
        return {}

    normalized = {}
    for key, value in names.items():
        try:
            normalized[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    return normalized


def normalize_2d(values: Any) -> list[list[float]]:
    data = to_plain_data(values)
    if not isinstance(data, list):
        return []
    if not data:
        return []
    if all(not isinstance(item, list) for item in data):
        return [data]
    return data


def to_plain_data(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return to_plain_data(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_data(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def safe_file_stem(text: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return stem or "curve"


def print_run_header(model_path: str, data_path: str, val_args: dict[str, Any], task: str) -> None:
    print("=" * 88)
    print("YOLO validation curve export")
    print(f"Task       : {task}")
    print(f"Model      : {model_path}")
    print(f"Data       : {data_path}")
    print(f"Split      : {val_args.get('split')}")
    print(f"Image size : {val_args.get('imgsz')}")
    print(f"Batch      : {val_args.get('batch')}")
    print(f"Output     : {val_args.get('project')} / {val_args.get('name')}")
    print("=" * 88)


if __name__ == "__main__":
    raise SystemExit(main())
