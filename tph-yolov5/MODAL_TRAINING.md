# TPH-YOLOv5 Modal-Style Training Parameters

This package uses the original YOLOv5-style `train.py` interface described in `README.md`: run options are passed as CLI flags, while loss and augmentation hyperparameters are passed through `--hyp`.

The command below mirrors the usable parts of the workspace `modal.yaml` for TPH-YOLOv5. YOLO26/newer-Ultralytics-only keys such as `use_wiou`, `use_nwd`, `use_repulsion`, `tal_topk*`, `select_mosaic`, `dfl`, `cutmix`, `copy_paste_mode`, `small_copy_paste_area`, `compile`, and `fraction` are not supported by this trainer and are intentionally omitted.

## Hyperparameter File

Use `data/hyps/hyp.modal.yaml`:

```yaml
lr0: 0.05
lrf: 0.05
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 5.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1

# YOLOv5 uses smaller box/object scalar conventions than newer Ultralytics.
box: 0.05
cls: 0.5
cls_pw: 1.0
obj: 1.0
obj_pw: 1.0
iou_t: 0.20
anchor_t: 4.0
fl_gamma: 0.0

hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
mosaic: 1.0
mixup: 0.0
copy_paste: 0.2
```

## Full Training Command

Run from the `tph-yolov5` directory:

```bash
python train.py \
  --img 640 \
  --batch-size 64 \
  --epochs 100 \
  --data /root/cqu/datasets/crowdhuman-dataset-people-and-faces-19k/data.yaml \
  --weights '' \
  --hyp data/hyps/hyp.modal.yaml \
  --cfg models/yolov5n-xs-tph.yaml \
  --device 0 \
  --workers 8 \
  --project ../juliet-heath/bysj \
  --name tph-yolov5-small-object \
  --patience 100 \
  --save-period -1
```

Notes:

- `--weights ''` trains from the YAML architecture. Do not reuse `modal.yaml`'s `yolo26n.pt`; it is a YOLO26 checkpoint, not a TPH-YOLOv5 checkpoint.
- `optimizer: SGD` is the default when `--adam` is omitted.
- `cos_lr: true` maps to the default YOLOv5 one-cycle scheduler; do not pass `--linear-lr`.
- `rect: false`, `resume: false`, `cache: false`, `single_cls: false`, and `multi_scale: 0.0` map to omitted flags because the YOLOv5 CLI enables those only when the flag is present.
- `val: true` maps to omitting `--noval`.

## COCO8 Smoke Test

Use a small batch and one epoch to verify that the entrypoint, model YAML, hyperparameters, and dataset parsing work:

```bash
python train.py \
  --img 320 \
  --batch-size 2 \
  --epochs 1 \
  --data /tmp/coco8-yolov5.yaml \
  --weights '' \
  --hyp data/hyps/hyp.modal.yaml \
  --cfg models/yolov5n-xs-tph.yaml \
  --device cpu \
  --workers 0 \
  --project ../runs/smoke \
  --name tph-yolov5-coco8 \
  --exist-ok \
  --nosave
```

On newer local environments, this older YOLOv5 code may need compatibility handling for NumPy 2.x and PyTorch 2.6+:

- `np.int` was removed in NumPy 2.x.
- PyTorch 2.6+ defaults `torch.load(..., weights_only=True)`, which can break old YOLOv5 checkpoint stripping/loading.
- Recent PyTorch is stricter about `Tensor.clamp_` scalar types in `utils/loss.py`.

The cleanest production fix is to use the dependency versions expected by this repository, or patch those legacy compatibility points in source before long training runs.
