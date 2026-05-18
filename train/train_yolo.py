"""
早期的 YOLO 2D 目标检测 (Object Detection) 训练代码，
用于输出只有位置信息的框选定位能力。
"""
import os
from ultralytics import YOLO

def main():
    print('[INFO] 加载模型中...')
    model = YOLO('yolov8n.pt')
    yaml_path = 'data/AppleBBCH81/data.yaml'
    if not os.path.exists(yaml_path):
        return
    print(f'[INFO] 使用数据集 {yaml_path} 开始训练...')
    model.train(data=yaml_path, 
                epochs=150, 
                imgsz=640, 
                batch=16, 
                device='0', 
                amp=False)
if __name__ == '__main__':
    main()

