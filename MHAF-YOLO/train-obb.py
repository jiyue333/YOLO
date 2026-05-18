from ultralytics import YOLO
if __name__ == '__main__':
    # model = YOLO('MAF-YOLOv2-s-obb.yaml')
    # model.train(data='DOTAv1-noms.yaml', batch=8, device=0, epochs=200, mixup=0.15, scale=0.9, degrees=180, flipud=0.5, imgsz=1024, workers=8, dfl=0.75)
    model = YOLO('MAF-YOLOv2-n-obb.yaml')
    model.train(data='DOTAv1-noms.yaml', batch=8, device=0, epochs=200, mixup=0.15, scale=0.9, degrees=180, flipud=0.5, imgsz=1024, workers=8, dfl=0.75)