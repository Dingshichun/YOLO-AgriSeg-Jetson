<div align="center">
  <h1>🍎 YOLO-AgriSeg-Jetson: 果园采摘机器人视觉感知系统</h1>
  <p>
    <b>简体中文</b> | <a href="README_en.md">English</a>
  </p>
  <p>从 YOLOv8+SAM 级联架构到全卷积 YOLO-seg 的终极演进，并突破边缘计算平台 (Jetson Nano B01) 物理部署极限的全栈式开源项目。</p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.6%20|%203.10-blue.svg" alt="Python">
    <img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg" alt="PyTorch">
    <img src="https://img.shields.io/badge/TensorRT-8.2+-76B900.svg" alt="TensorRT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  </p>
</div>

## 🎥 效果展示 (Visualizations)

<div align="center">
  <img src="visualization/imgsz320.gif" width="80%" alt="Jetson Edge 端极限推理">
  <br>
  <sup>Jetson Nano 原生 TensorRT imgsz=320 实时推流渲染展示</sup>
  <br><br>
  <img src="visualization/gui_control_station.png" width="80%" alt="上位机GUI工作站展示">
  <br>
  <sup>跨平台上位机实时 Conf 滑动条动态反馈 GUI</sup>
  <br><br>
  <img src="visualization/imgsz320_test.png" width="80%" alt="Jetson Edge 端极限推理截屏">
  <br>
  <sup>分辨率极速化 imgsz=320 模式下边缘侧画面</sup>
</div>

## 🌟 核心技术与亮点 (Key Features)

> 📖 **深度阅读**：关于本项目如何解决机器人边缘算力瓶颈，以及为何从 YOLO+SAM 级联架构迁移至单体 YOLO-seg 的详细心路历程与技术取舍，请参阅 [架构演进与技术 (Pipeline Evolution Analysis)](pipeline_evolution_analysis.md)。

- **🤖 零成本像素级标注 (Auto-Annotation)**
  - 彻底消除极其高昂的人工多边形 (Polygon) 标注成本。利用 **SAM (Segment Anything)** 进行一次性离线处理，把单纯的矩形框 (BBox) 自动蒸馏转化为原生 YOLOv8-seg 训练所需的多边形掩码。
- **⚡ 架构进化 (YOLOv8n+SAM ⟶ YOLOv8n-seg)**
  - 废弃了速度极慢的双模型 Prompt 级联管线。通过重训端到端的单体 YOLOv8n-seg 模型，在苹果数据集上实现 `mAP50 96.26%` 高精度，且推理复杂度骤降为 `O(1)`。
- **💻 跨平台原生 GUI 工作站 (Tkinter + OpenCV)**
  - 拒绝臃肿的 PyQt / Web 框架，使用 Python 纯原生标准库打造极低开销的上位机，支持导入图片、视频和连接本地摄像头，并且实现了**滑动条动态刷新置信度掩码**。
- **📊 MOT 多目标轨迹产量清点 (ByteTrack)**
  - 辅以 `lapx` 和虚拟越界线 (Virtual Crossing Line) 判定技术，支持机器人边走边算，提供果园流水线产量去重估算。
- **☠️ 极限边缘部署 (Jetson Nano B01)**
  - 适配硬件（Maxwell 核心，JetPack 4.6.1, Python 3.6.9）。不使用 INT8 制造负优化，锁定 `FP16` + `opset=12` 兼容黄金范式。
  - **彻底脱离臃肿的 `ultralytics` 等高级库依赖**。构建了纯 `PyCUDA` + `TensorRT` API 的高速数据通道，挂载原生 GStreamer ISP 处理 CSI 读入导致的底层绿屏乱码。手工解包 C++ 张量矩阵、设计 Numpy NMS 拆解器，imgsz=320 时稳定跑在 30 FPS 以上。

## 📂 项目目录结构 (Project Structure)

```text
YOLO-AgriSeg-Jetson/
├── tools/                  # 辅助工具模块与应用演示
│   ├── auto_annotate_all.py# 全自动刷库标注脚本 (BBox -> Polygon)
│   ├── gui_control_station.py # Windows/Linux 跨平台上位机操作台
│   └── track_and_count.py  # ByteTrack 苹果产量视频边界推流计数
├── train/                  # 模型训练、推断验证与常规模型格式导出
│   ├── main_pipeline.py    # 早期 YOLO+SAM 双架构测试流水线
│   ├── main_yolo_seg.py    # YOLOv8-seg 单体验证推断逻辑
│   ├── train_yolo.py       # 常规 YOLO 初始目标检测训练
│   ├── train_yolo_seg.py   # YOLO-seg 网络核心微调脚本
│   ├── export_tensorrt.py  # 基础检测模型导出器
│   └── export_yolo_seg_tensorrt.py # 分割级模型原生态导出接口
├── deploy/                 # 核心边缘终端开发板（Jetson）部署
│   ├── build_engine.sh     # Jetson Nano 本地 trtexec 编译引擎入口
│   ├── export_jetson_tensorrt.py # 提前生成 opset=12 的强兼容态导出
│   └── run_jetson_tensorrt.py # Jetson 脱离 Ultralytics C++-Binding 高速裸推理
├── visualization/          # 截图、动图素材库
├── data/                   # 包含原始盒框与重构成 Polygon 的数据集
├── summary.md              # 【避坑指南】架构演化全纪实与原理全解
├── requirements.txt        # PC / 开发板所需独立依赖包
└── README.md               # 项目主说明文档
```

## 📦 数据集 (Dataset)
本项目所使用的苹果果园数据集通过以下链接下载：
👉 **[Apple Orchards Vision Dataset (AppleBBCH81)](https://www.kaggle.com/datasets/projectlzp201910094/applebbch81)**

原始数据集仅包含目标检测的矩形框（Bounding Box）标注，你可以使用本项目提供的 `tools/auto_annotate_all.py` 脚本，利用 SAM 自动生成用于训练 YOLO-seg 的多边形（Polygon）掩码标注。

## 🚀 快速开始 (Quick Start)

### 1. 安装上位机环境
```bash
git clone https://github.com/Dingshichun/YOLO-AgriSeg-Jetson.git
cd YOLO-AgriSeg-Jetson
conda create -n agrivision python==3.10
conda activate agrivision
pip install -r requirements.txt
```
*注：`lapx` 等追踪依赖包含其中，但 `pycuda` 与 `tensorrt` 为 Jetson 绑定，按需编译。*

### 2. 运行交互式控制站 (GUI)
如果想在开发机或具有桌面环境的机器人端体验零延迟推理调参：
```bash
python gui_control_station.py
```

### 3. 数据全自动标注 (使用 SAM)
修改 `tools/auto_annotate_all.py` 中的输入数据集路径，即刻为只有普通纯 Box 框标注的野生数据集生成配套的实例分割 Mask Polygon TXT 文本：
```bash
python tools/auto_annotate_all.py
```

## 🧱 Jetson Nano 部署总结 (Deployment)
要让落后的边缘板以 20+ FPS 流畅跑通最新的分割模型，请详细阅读 [`summary.md`](summary.md) 及 `deploy/run_jetson_tensorrt.py` 中的详尽注释。本项目成功跨越了以下技术鸿沟：
1. **OOM 与 OpSet 不兼容**：开发机侧导出时强锁定 `opset=12`, `workspace=2048`, `imgsz=416`。
2. **量化反向降速坑**：规避软解 INT8 陷阱，使用架构满血硬件原生搭载的 FP16。
3. **CSI 硬件 ISP 强拉屏**：规避 Linux V4L2 协议 YUYV 空满载绿屏及 Bayer 去马赛克失败，以 GStreamer + `nvarguscamerasrc` 建流解码。
4. **解包阵列穿透**：底层解耦 TensorRT 原生推理与 C++ 张量结构，徒手写 Numpy 数据映射转换桥重构 NMS 检出逻辑。

## 📄 分发协议
本项目采用 [MIT License](LICENSE) 协议进行分发。欢迎交流或二次投入开发落地。