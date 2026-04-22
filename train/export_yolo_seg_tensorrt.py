"""
将训练好的 YOLOv8-seg 实例分割模型直接导出为常规的 TensorRT 引擎，用于推理测试。
实际部署时需要在目标设备（如 Jetson Nano/Orin）上进行导出，
以确保生成的 .engine 文件与设备的 GPU 和 TensorRT 版本兼容。
"""
import os
from ultralytics import YOLO

def main():
    print("[INFO] 开始 YOLOv8-seg 模型的 TensorRT 导出流程...")
    
    # 训练后得到的分割模型权重路径
    model_path = "runs/segment/train/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"[WARN] 找不到模型权重文件: {model_path}")
        print("[INFO] 如果还没有完成最后这步多边形数据的训练，请先运行完毕 python train_yolo_seg.py")
        print("[INFO] 若为了测试此脚本，可临时将 model_path 改为 'yolov8n-seg.pt'")
        return
        
    print(f"[INFO] 正在加载分割模型: {model_path}")
    # 会自动识别出这是一个分割模型
    model = YOLO(model_path)
    
    print("[INFO] 正在导出为 TensorRT (.engine) 格式...")
    
    # 核心导出参数说明：
    # format="engine": 指定导出 TensorRT 引擎
    # half=True: 启用 FP16 半精度量化（在基本不掉点的前提下，大幅增加显存吞吐，是部署的标配）
    # imgsz=640: 由于 TensorRT 默认是静态计算图最佳，输入尺寸最好锁死
    # simplify=True: 在转换过程中自动简化图结构
    export_path = model.export(
        format="engine",
        half=True,
        imgsz=640,
        device="0",
        simplify=True
    )
    
    print(f"\n✅ 导出成功！TensorRT Engine 模型已保存。")
    print(f"    生成路径大致为: {export_path} (默认同目录下 .engine 文件)")
    print("-" * 50)
    print("⚠️ 重要部署提示 ：")
    print("1. TensorRT 的 .engine 文件与生成它的【显卡型号】和【TensorRT版本】是绝对强绑定的。")
    print("2. 也就是说，使用台式机导出的 .engine 不能直接拷贝到机器人的大臂端(比如 Nvidia Jetson)上跑，会直接报错。")
    print("3. 正确做法是：把 'best.pt' 拷贝到真的机器人上，然后在机器人的终端里再跑一次本脚本，将其转化为设备专用的 engine！")

if __name__ == "__main__":
    main()