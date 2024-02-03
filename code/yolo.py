import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    #model = YOLO('ultralytics/cfg/models/v3/yolov3-tiny.yaml')
    #model = YOLO('ultralytics/cfg/models/v8/yolov8.yaml')
    model = YOLO('ultralytics/cfg/models/v5/yolov5.yaml')
    #model.load('yolov3-tinyu.pt') # loading pretrain weights
    model.load('yolov5n.pt')  # loading pretrain weights
    model.train(data='dataset/data.yaml',
                cache=False,
                imgsz=320,
                #imgsz=640,
                epochs=100,
                batch=70,
                close_mosaic=10,
                workers=4,
                device='0',
                optimizer='SGD', # using SGD
                # resume='', # last.pt path
                # amp=False # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                lr0=0.28,
                lrf= 0.28,
    )