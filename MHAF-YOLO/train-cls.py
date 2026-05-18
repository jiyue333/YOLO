from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('MAF-YOLOv2-n-cls.yaml')
    model.train(data='../datasets/imagenet', batch=256, device=0, epochs=300, scale=0.5, imgsz=224, workers=8,
                cos_lr=True, lr0=0.1, momentum=0.9, weight_decay=0.0001, warmup_epochs=0, hsv_s=0.4)
