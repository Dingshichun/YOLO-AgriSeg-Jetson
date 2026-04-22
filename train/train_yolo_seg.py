"""
YOLOv8-seg 实例分割模型的训练脚本，
加载 -seg 权重并在已被 SAM 洗印处理好的包含 Mask Txt 标注的 Apple 训练集上实现 O(1) 前向推理的模型蒸馏。
"""
import os
from ultralytics import YOLO

def main():
    print('[INFO] 加载模型中... 准备进行实例分割 (Segmentation) 训练')
    
    # 加载带有 "-seg" 后缀的官方预训练分割权重
    model = YOLO('yolov8n-seg.pt')
    
    yaml_path = 'data/AppleBBCH81/data.yaml'
    
    if not os.path.exists(yaml_path):
        print(f"[ERROR] 找不到此数据集文件: {yaml_path}")
        return
        
    print(f'[INFO] 使用包含全新 polygon 标签的数据集 {yaml_path} 开始训练...')
    
    # 明确指定 task="segment"，启动端到端的实例分割训练
    model.train(
        data=yaml_path, 
        epochs=150, 
        imgsz=640, 
        batch=16, 
        device='0', 
        amp=False, # 关闭混合精度训练以确保训练过程的稳定性，尤其是在资源有限的环境中
        task='segment'
    )
    
    print("[INFO] 训练结束！新模型最佳权重保存在类似 runs/segment/train/weights/best.pt 的目录下。")
    print("[INFO] 可以直接使用这个单独的 engine 抛弃 SAM 进行单模型实时检测+分割！")

if __name__ == '__main__':
    main()