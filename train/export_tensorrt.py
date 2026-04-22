"""
将训练好的初期 YOLOv8 目标检测模型导出为高精度、加速的 RT 引擎文件用于推理测试。
"""
import os
from ultralytics import YOLO

def export_model_to_tensorrt(model_path="models/weights/best.pt"):
    """
    将训练好的 PyTorch 模型 (.pt) 导出为 TensorRT 引擎 (.engine)，
    用于提升在 NVIDIA GPU（如用于农业机器人的 Jetson Nano/Orin）上的实时推理帧率。
    这里只是测试性地导出为 TensorRT，实际部署时需要在目标设备上进行导出以确保兼容性。
    """
    print(f"[INFO] 正在加载 PyTorch 模型: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"[ERROR] 找不到模型文件: {model_path}")
        print("💡 请先完成 YOLO 训练并生成 best.pt，或修改为实际的权重路径 (如 runs/detect/train/weights/best.pt)。")
        return

    # 加载已训练的模型
    model = YOLO(model_path)

    print("[INFO] 开始导出为 TensorRT 格式 (.engine) ...")
    print("⚠️  注意: 导出过程需要安装 tensorrt 库 (pip install tensorrt)，并且耗时可能需要几分钟，请耐心等待。")
    print("⚠️  在这期间会进行计算图融合 (Layer Fusion) 和精度量化。")

    # 核心导出代码：
    # format="engine": 目标格式为 TensorRT
    # fp16=True: 采用半精度量化，极大提升速度，精度损失极小，非常符合自动驾驶和机器人视觉场景
    # dynamic=True: 支持推理时动态的 batch size 变化
    # simplify=True: 简化并优化 ONNX（open neural network exchange, 开放神经网络交换格式）计算图结构
    try:
        exported_path = model.export(
            format="engine",
            half=True,      # 默认开启 FP16 量化加速
            dynamic=True,   # 允许动态输入张量大小
            simplify=True,
            workspace=4     # 给 TensorRT 分配最大 4GB 显存用于编译优化网络
        )
        print(f"\n[SUCCESS] 模型已成功导出！")
        print(f"[SUCCESS] TensorRT 引擎文件路径: {exported_path}")
        print("\n[NEXT STEP] 在 main_pipeline.py 中，可以直接把权重路径换成这个 .engine 文件：")
        print("  yolo_weight = 'models/weights/best.engine'")
        print("  yolo_model = YOLO(yolo_weight)  # 代码无需修改，Ultralytics 会自动调用 TensorRT Runtime 执行推理！")
        
    except Exception as e:
        print(f"\n[ERROR] 导出失败，原因: {e}")
        print("请检查是否已正确安装 NVIDIA TensorRT 环境。")

if __name__ == "__main__":
    # 你可以替换为你实际训练完的 best.pt 路径
    # 我的训练输出在 runs/segment/train/weights/best.pt，
    # 这里我又把它复制了一份到 models/weights/best.pt 以便管理
    target_weight_path = "models/weights/best.pt" 
    export_model_to_tensorrt(target_weight_path)
