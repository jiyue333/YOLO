from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import yaml

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ULTRALYTICS_REPO = WORKSPACE_ROOT / "ultralytics-main"
LOCAL_ULTRALYTICS_PACKAGE = LOCAL_ULTRALYTICS_REPO / "ultralytics"
# 
SECTION_TO_KEYS: dict[str, tuple[str, ...]] = {
    "experiment": (
        "task",
        "mode",
        "project",
        "name",
        "exist_ok",
        "verbose",
        "seed",
        "deterministic",
    ),
    "model": (
        "model",
        "pretrained",
        "freeze",
        "compile",
    ),
    "dataset": (
        "data",
        "fraction",
        "single_cls",
    ),
    "train": (
        "epochs",
        "time",
        "patience",
        "batch",
        "imgsz",
        "save",
        "save_period",
        "cache",
        "device",
        "workers",
        "optimizer",
        "rect",
        "cos_lr",
        "close_mosaic",
        "resume",
        "amp",
        "ema_decay",
        "profile",
        "multi_scale",
    ),
    "segmentation": (
        "overlap_mask",
        "mask_ratio",
    ),
    "classification": (
        "dropout",
    ),
    "validation": (
        "val",
        "split",
        "save_json",
        "conf",
        "iou",
        "max_det",
        "nms_max_time_img",
        "half",
        "dnn",
        "plots",
        "end2end",
        "visualize",
    ),
    "prediction": (
        "source",
        "vid_stride",
        "stream_buffer",
        "inference_end2end",
        "sahi",
        "sahi_slice_size",
        "sahi_overlap",
        "adaptive_nms",
        "adaptive_nms_max_iou",
        "adaptive_nms_density_threshold",
        "inference_augment",
        "augment",
        "agnostic_nms",
        "classes",
        "retina_masks",
        "embed",
    ),
    "visualize": (
        "show",
        "save_frames",
        "save_txt",
        "save_conf",
        "save_crop",
        "show_labels",
        "show_conf",
        "show_boxes",
        "line_width",
    ),
    "export": (
        "format",
        "keras",
        "optimize",
        "int8",
        "dynamic",
        "simplify",
        "opset",
        "workspace",
        "nms",
    ),
    "hyperparameters": (
        "lr0",
        "lrf",
        "momentum",
        "weight_decay",
        "warmup_epochs",
        "warmup_momentum",
        "warmup_bias_lr",
        "box",
        "cls",
        "cls_pw",
        "dfl",
        "pose",
        "kobj",
        "rle",
        "angle",
        "nbs",
        "wiou_alpha",
        "wiou_delta",
        "wiou_momentum",
        "nwd_weight",
        "nwd_small_area",
        "nwd_constant",
        "repgt_weight",
        "repulsion_sigma",
        "tal_topk",
        "tal_topk_one2one",
        "tal_topk_one2one_secondary",
        "cls_pos_weight",
    ),
    "augmentations": (
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "flipud",
        "fliplr",
        "bgr",
        "mosaic",
        "mixup",
        "cutmix",
        "copy_paste",
        "copy_paste_mode",
        "small_copy_paste_area",
        "select_mosaic",
        "motion_blur",
        "cached_mixup",
        "mixup_cache_size",
        "auto_augment",
        "erasing",
    ),
    "tracker": (
        "tracker",
    ),
}

META_SECTIONS = {"meta"}
RAW_SECTIONS = {"raw", "custom"}
VALID_YOLO_KEYS = {key for keys in SECTION_TO_KEYS.values() for key in keys} | {"cfg"}
TOP_LEVEL_HINTS = set(SECTION_TO_KEYS) | META_SECTIONS | RAW_SECTIONS | VALID_YOLO_KEYS
PATH_LIKE_KEYS = {"model", "pretrained", "data", "project", "cfg", "resume", "source", "tracker"}


class ConfigError(ValueError):
    """Raised when the experiment configuration is invalid."""


@dataclass(slots=True)
class PreparedRun:
    """Normalized experiment payload ready to be passed to Ultralytics."""

    config_path: Path
    declared_mode: Any
    yolo_args: dict[str, Any]


def run_from_cli(mode: str) -> int:
    """Parse CLI arguments and run the requested mode."""
    parser = build_parser(mode)
    args = parser.parse_args()
    prepared = prepare_run(mode=mode, cli_args=args)
    print_summary(prepared=prepared, mode=mode, show_config=args.show_config or args.dry_run, dry_run=args.dry_run)

    if args.dry_run:
        print(f"\n[dry-run] 已完成 {mode} 配置解析，不会真正调用 Ultralytics。")
        return 0

    model = build_model(prepared.yolo_args["model"], prepared.yolo_args.get("task"))
    runtime_args = prepared.yolo_args.copy()
    runtime_args.pop("mode", None)

    if mode == "train":
        model.train(**runtime_args)
    elif mode == "val":
        model.val(**runtime_args)
    else:
        raise ConfigError(f"暂不支持的模式: {mode}")

    return 0


def build_parser(mode: str) -> argparse.ArgumentParser:
    """Create a friendly CLI for the training/validation wrapper."""
    examples = {
        "train": (
            "示例:\n"
            "  python script/train.py --config default.yaml\n"
            "  python script/train.py -c default.yaml --epochs 300 --batch 8\n"
            "  python script/train.py -c default.yaml --set train.epochs=300 --set hyperparameters.lr0=0.005"
        ),
        "val": (
            "示例:\n"
            "  python script/val.py --config default.yaml\n"
            "  python script/val.py -c default.yaml --model runs/experiments/exp/weights/best.pt\n"
            "  python script/val.py -c default.yaml --split test --save-json"
        ),
    }
    parser = argparse.ArgumentParser(
        description=f"友好的 Ultralytics {mode} 实验封装脚本",
        epilog=examples[mode],
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        default=str(WORKSPACE_ROOT / "default.yaml"),
        help="实验配置文件路径，默认读取工作区根目录 default.yaml",
    )
    parser.add_argument("--model", "--weights", dest="model", default=None, help="覆盖模型权重或模型结构 yaml")
    parser.add_argument("--data", default=None, help="覆盖数据集 yaml")
    parser.add_argument("--project", default=None, help="覆盖输出目录根路径")
    parser.add_argument("--name", default=None, help="覆盖实验名称")
    parser.add_argument(
        "--task",
        choices=["detect", "segment", "classify", "pose", "obb"],
        default=None,
        help="覆盖任务类型",
    )
    parser.add_argument("--device", default=None, help="覆盖设备，例如 0 / cpu / mps / 0,1")
    parser.add_argument("--imgsz", default=None, help="覆盖图像尺寸，例如 640 或 [640, 640]")
    parser.add_argument("--batch", default=None, help="覆盖 batch，大于 1 为固定 batch，0~1 可用于 AutoBatch")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="用点路径覆盖任意配置，例如 train.epochs=300 或 raw.cfg='custom.yaml'",
    )
    parser.add_argument("--show-config", action="store_true", help="打印最终展开后的 Ultralytics 参数")
    parser.add_argument("--dry-run", action="store_true", help="只解析配置并打印结果，不真正执行")

    if mode == "train":
        parser.add_argument("--epochs", default=None, help="覆盖训练轮数")
        parser.add_argument(
            "--resume",
            nargs="?",
            const=True,
            default=None,
            help="恢复训练；不带值时等价于 True，也可以直接传 last.pt 路径",
        )
        parser.add_argument("--optimizer", default=None, help="覆盖优化器，例如 auto / SGD / AdamW")
        parser.add_argument("--lr0", default=None, help="覆盖初始学习率")
    elif mode == "val":
        parser.add_argument("--conf", default=None, help="覆盖验证置信度阈值")
        parser.add_argument("--split", choices=["train", "val", "test"], default=None, help="覆盖验证数据划分")
        parser.add_argument("--save-json", action="store_true", default=None, help="保存 COCO JSON 评估结果")
        parser.add_argument("--half", action="store_true", default=None, help="启用 FP16 验证")

    return parser


def prepare_run(mode: str, cli_args: argparse.Namespace) -> PreparedRun:
    """Load, validate, flatten, and override an experiment config."""
    config_path = Path(cli_args.config).expanduser().resolve()
    config = load_config(config_path)
    apply_cli_set_overrides(config, cli_args.set)

    flattened = flatten_config(config)
    declared_mode = flattened.get("mode")
    flattened.update(build_direct_overrides(mode=mode, cli_args=cli_args))
    flattened["mode"] = mode
    flattened = resolve_known_paths(flattened, config_path=config_path)

    if not flattened.get("model"):
        raise ConfigError("缺少模型配置，请在 default.yaml 的 model.model 中指定，或通过 --model 覆盖。")

    return PreparedRun(config_path=config_path, declared_mode=declared_mode, yolo_args=flattened)


def load_config(config_path: Path) -> dict[str, Any]:
    """Read a YAML experiment config from disk."""
    if not config_path.exists():
        raise ConfigError(f"找不到配置文件: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ConfigError("配置文件顶层必须是一个 YAML 字典对象。")

    return data


def apply_cli_set_overrides(config: dict[str, Any], overrides: list[str]) -> None:
    """Apply dotted CLI overrides to the raw config tree before flattening."""
    for entry in overrides:
        key, value = parse_key_value(entry)
        set_nested_value(config, key.split("."), value)


def flatten_config(config: dict[str, Any]) -> dict[str, Any]:
    """Flatten grouped experiment YAML into native Ultralytics args."""
    flattened: dict[str, Any] = {}

    for top_key, value in config.items():
        canonical_key = "raw" if top_key in RAW_SECTIONS else top_key

        if canonical_key in META_SECTIONS:
            continue

        if canonical_key == "raw":
            if value is None:
                continue
            if not isinstance(value, dict):
                raise ConfigError("raw/custom 段必须是一个字典，例如 raw: {cfg: null}。")
            ensure_known_keys(scope="raw", provided_keys=value.keys(), allowed_keys=VALID_YOLO_KEYS)
            flattened.update(value)
            continue

        if canonical_key in SECTION_TO_KEYS and isinstance(value, dict):
            ensure_known_keys(scope=canonical_key, provided_keys=value.keys(), allowed_keys=SECTION_TO_KEYS[canonical_key])
            flattened.update(value)
            continue

        if top_key in VALID_YOLO_KEYS:
            flattened[top_key] = value
            continue

        if canonical_key in SECTION_TO_KEYS:
            raise ConfigError(
                f"配置段 '{top_key}' 需要写成字典形式，例如:\n"
                f"{top_key}:\n  {SECTION_TO_KEYS[canonical_key][0]}: <value>"
            )

        raise_unknown_key_error(scope="顶层", key=top_key, allowed_keys=TOP_LEVEL_HINTS)

    return flattened


def ensure_known_keys(scope: str, provided_keys: Any, allowed_keys: Any) -> None:
    """Raise a friendly error when a section contains unknown keys."""
    unknown_keys = sorted(set(provided_keys) - set(allowed_keys))
    if not unknown_keys:
        return

    messages = []
    for key in unknown_keys:
        suggestion = suggest_key(key, allowed_keys)
        messages.append(f"- [{scope}] 未知字段 '{key}'{suggestion}")
    raise ConfigError("\n".join(messages))


def raise_unknown_key_error(scope: str, key: str, allowed_keys: set[str]) -> None:
    """Raise a consistent unknown-key error with suggestions."""
    suggestion = suggest_key(key, allowed_keys)
    raise ConfigError(f"[{scope}] 未知字段 '{key}'{suggestion}")


def suggest_key(key: str, allowed_keys: Any) -> str:
    """Suggest close matches for mistyped config keys."""
    matches = get_close_matches(key, sorted(allowed_keys), n=3, cutoff=0.55)
    return f"，你也许想写: {', '.join(matches)}" if matches else ""


def set_nested_value(target: dict[str, Any], path: list[str], value: Any) -> None:
    """Set a nested config value using dotted-path syntax."""
    current = target
    for token in path[:-1]:
        if token in RAW_SECTIONS:
            token = "raw"
        existing = current.get(token)
        if existing is None:
            current[token] = {}
        elif not isinstance(existing, dict):
            raise ConfigError(f"无法覆盖 '{'.'.join(path)}'：'{token}' 当前不是字典。")
        current = current[token]
    current[path[-1]] = value


def build_direct_overrides(mode: str, cli_args: argparse.Namespace) -> dict[str, Any]:
    """Collect non-dotted CLI overrides."""
    overrides: dict[str, Any] = {}
    field_names = ["model", "data", "project", "name", "task", "device", "imgsz", "batch"]
    if mode == "train":
        field_names.extend(["epochs", "resume", "optimizer", "lr0"])
    elif mode == "val":
        field_names.extend(["conf", "split", "save_json", "half"])

    for field_name in field_names:
        value = getattr(cli_args, field_name, None)
        if value is not None:
            overrides[field_name] = smart_value(value) if isinstance(value, str) else value

    return overrides


def resolve_known_paths(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    """Resolve common path-like fields relative to the config file when possible."""
    resolved = config.copy()
    for key in PATH_LIKE_KEYS:
        if key in resolved:
            resolved[key] = resolve_path_value(key=key, value=resolved[key], config_path=config_path)
    return resolved


def resolve_path_value(key: str, value: Any, config_path: Path) -> Any:
    """Resolve a path-like string to an absolute path when it clearly refers to a local file."""
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return value

    if key == "source" and (
        text.isdigit()
        or text.startswith(("http://", "https://", "rtsp://", "rtmp://", "tcp://", "ul://"))
    ):
        return value

    path = Path(text).expanduser()
    if path.is_absolute():
        return str(path)

    if key == "project":
        return str((config_path.parent / path).resolve())

    candidates = [config_path.parent / path, WORKSPACE_ROOT / path]

    if key == "data":
        candidates.append(LOCAL_ULTRALYTICS_PACKAGE / "cfg" / "datasets" / path.name)
    elif key == "tracker":
        candidates.append(LOCAL_ULTRALYTICS_PACKAGE / "cfg" / "trackers" / path.name)
    elif key == "cfg":
        candidates.append(LOCAL_ULTRALYTICS_PACKAGE / "cfg" / path.name)
    elif key == "model" and path.suffix in {".yaml", ".yml"}:
        matches = list((LOCAL_ULTRALYTICS_PACKAGE / "cfg" / "models").rglob(path.name))
        if len(matches) == 1:
            candidates.append(matches[0])

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    return value


def print_summary(prepared: PreparedRun, mode: str, show_config: bool = False, dry_run: bool = False) -> None:
    """Print a concise run summary for human-friendly experiment tracking."""
    args = prepared.yolo_args
    print("=" * 88)
    print(f"YOLO 实验封装 | mode={mode}")
    print(f"配置文件      : {prepared.config_path}")
    print(f"本地源码仓库  : {LOCAL_ULTRALYTICS_REPO}")
    print(f"任务类型      : {args.get('task')}")
    print(f"模型          : {args.get('model')}")
    print(f"数据集        : {args.get('data')}")
    print(f"输出目录      : {args.get('project')} / {args.get('name')}")

    if mode == "train":
        print(
            "训练参数      : "
            f"epochs={args.get('epochs')} | batch={args.get('batch')} | imgsz={args.get('imgsz')} | device={args.get('device')}"
        )
        print(f"恢复训练      : {args.get('resume')}")
    else:
        print(
            "验证参数      : "
            f"split={args.get('split')} | conf={args.get('conf')} | iou={args.get('iou')} | half={args.get('half')}"
        )

    if prepared.declared_mode is not None and prepared.declared_mode != mode:
        print(f"配置中的 mode : {prepared.declared_mode}（已由脚本强制切换为 {mode}）")

    if show_config:
        print("-" * 88)
        print("最终 Ultralytics 参数：")
        print(yaml.safe_dump(args, sort_keys=False, allow_unicode=True))

    if dry_run:
        print("-" * 88)


def build_model(model_value: str, task: str | None = None):
    """Build an Ultralytics model using the local source tree first."""
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
    """Make sure the local Ultralytics repo is imported before any installed wheel."""
    if not LOCAL_ULTRALYTICS_REPO.exists():
        raise ConfigError(f"未找到本地 Ultralytics 仓库: {LOCAL_ULTRALYTICS_REPO}")
    repo_text = str(LOCAL_ULTRALYTICS_REPO.resolve())
    for existing in (repo_text, str(LOCAL_ULTRALYTICS_REPO)):
        while existing in sys.path:
            sys.path.remove(existing)
    sys.path.insert(0, repo_text)

    loaded_module = sys.modules.get("ultralytics")
    if loaded_module is None:
        return

    module_file = getattr(loaded_module, "__file__", None)
    if module_file is None:
        raise ConfigError("当前进程已导入未知来源的 ultralytics，无法确认是否为本地源码。")
    try:
        Path(module_file).resolve().relative_to(LOCAL_ULTRALYTICS_PACKAGE.resolve())
    except ValueError as exc:
        raise ConfigError(
            "当前进程已从非本地源码导入 ultralytics，"
            f"实际路径: {module_file}；期望路径位于: {LOCAL_ULTRALYTICS_PACKAGE}"
        ) from exc


def parse_key_value(entry: str) -> tuple[str, Any]:
    """Parse KEY=VALUE text into a key and a smart-casted value."""
    if "=" not in entry:
        raise ConfigError(f"覆盖参数 '{entry}' 缺少 '='，正确格式例如 train.epochs=300")
    key, raw_value = entry.split("=", 1)
    key = key.strip()
    raw_value = raw_value.strip()
    if not key:
        raise ConfigError(f"覆盖参数 '{entry}' 缺少键名")
    if raw_value == "":
        raise ConfigError(f"覆盖参数 '{entry}' 缺少值")
    return key, smart_value(raw_value)


def smart_value(value: Any) -> Any:
    """Convert common CLI strings into Python objects."""
    if not isinstance(value, str):
        return value

    lowered = value.lower()
    if lowered == "none":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value
