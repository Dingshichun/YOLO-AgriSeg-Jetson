#!/bin/bash

echo "开始使用 trtexec 编译 TensorRT Engine..."
echo "此过程在 Jetson Nano 上可能需要 15-20 分钟，请耐心等待！"

# 修改 onnx 文件路径和输出 engine 文件路径，确保和实际文件位置一致
/usr/src/tensorrt/bin/trtexec \
    --onnx=best.onnx \
    --saveEngine=best.engine \
    --fp16 \
    --workspace=2048

echo "引擎编译完成！"
echo "现在可以在 Jetson Nano 上使用该引擎进行高效的推理了！"
