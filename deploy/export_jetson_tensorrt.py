"""
专用于在上位机/开发机上，以 opset=12 和 simplify=True 将 pytorch 格式的模型转换为 ONNX 格式，为 Jetson Nano B01 设备(Jetpack 4.6.1)做兼容性准备。
"""
import os
from ultralytics import YOLO

def export_to_onnx():
    # 替换为自己训练好的 best.pt 的路径
    pt_path = "runs/segment/train/weights/best.pt"
    
    if not os.path.exists(pt_path):
        print(f"找不到 {pt_path}，请检查路径。默认使用 yolov8n-seg.pt")
        pt_path = "yolov8n-seg.pt"

    # 加载 PyTorch 模型
    model = YOLO(pt_path, task='segment')

    print(f"正在准备将 {pt_path} 转换为 ONNX 格式...")
    print("注意: 针对 JetPack 4.6.1 (包含 TensorRT 8.2)，已指定 opset=12 和 simplify=True")

    # 执行导出
    # -- format="onnx": 导出为 ONNX 格式
    # -- opset=12: 兼容 JetPack 4.6.1 环境，避免不支持高于 12 的算子
    # -- simplify=True: 简化模型网络结构，移除冗余算子，增强兼容性
    model.export(
        format="onnx",
        opset=12,
        simplify=True,
        imgsz=640        # 如果 Nano 跑 640 卡顿，改为 416 或 320 重新导出
    )

    print("导出完毕！当前目录下应生成了对应的 .onnx 文件。")
    print("\n【下一步引导】")
    print("1. 请将生成的 .onnx 文件拷贝至 Jetson Nano。")
    print("2. 在 Nano 的终端中执行下面的命令，将其编译为 engine 引擎（此过程需10-20分钟）：")
    print("   /usr/src/tensorrt/bin/trtexec --onnx=best.onnx --saveEngine=best.engine --fp16 --workspace=2048")
    # 我是将上面的命令写成了一个 shell 脚本 build_engine.sh
    # 需要添加可执行权限：chmod +x build_engine.sh，Nano 端直接执行 bash build_engine.sh 就行了
if __name__ == "__main__":
    export_to_onnx()
