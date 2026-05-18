from ultralytics import YOLOv10

if __name__ == '__main__':
    model = YOLOv10('MAF-YOLOv10-n.pt')
    model.val(data='coco.yaml', device=0,split='val', save_json=True, batch=16)
