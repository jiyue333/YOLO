from pathlib import Path

from ultralytics import YOLOv10


MHAF_ROOT = Path(__file__).resolve().parent
MODEL_CFG = "/root/cqu/YOLO/MHAF-YOLO/ultralytics/cfg/models/v10/yolov10n-mafpn.yaml"
DATA_CFG = "/root/cqu/datasets/crowdhuman-dataset-people-and-faces-19k/data.yaml"

# Values mirrored from modal.yaml, limited to keys supported by MHAF-YOLO's bundled Ultralytics config.
TRAIN_ARGS = {
    # Experiment / dataset
    "data": DATA_CFG,
    "project": "juliet-heath/bysj",
    "name": "yolo-26n-small-object",
    "exist_ok": False,
    "verbose": True,
    "seed": 0,
    "deterministic": False,
    "fraction": 0.2,
    "single_cls": False,
    # Train
    "epochs": 100,
    "time": None,
    "patience": 100,
    "batch": 64,
    "imgsz": 640,
    "save": True,
    "save_period": -1,
    "cache": False,
    "device": 0,
    "workers": 8,
    "optimizer": "SGD",
    "rect": False,
    "cos_lr": True,
    "close_mosaic": 10,
    "resume": False,
    # Disabled for PyTorch 2.6+ compatibility: the bundled AMP check loads
    # yolov8n.pt through legacy torch.load semantics.
    "amp": False,
    "profile": False,
    "multi_scale": False,
    # Validation during training
    "val": True,
    "split": "val",
    "save_json": False,
    "conf": 0.01,
    "iou": 0.7,
    "max_det": 600,
    "half": False,
    "dnn": False,
    "plots": True,
    # Loss and optimizer hyperparameters
    "lr0": 0.05,
    "lrf": 0.05,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 5.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "box": 5.0,
    "cls": 0.5,
    "dfl": 1.5,
    "pose": 12.0,
    "kobj": 1.0,
    "nbs": 64,
    # Augmentations
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "bgr": 0.0,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.2,
    "auto_augment": "randaugment",
    "erasing": 0.4,
}


if __name__ == "__main__":
    model = YOLOv10(str(MODEL_CFG))
    model.train(**TRAIN_ARGS)
