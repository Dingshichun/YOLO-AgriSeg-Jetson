# YOLO-AgriSeg-Jetson: 从 YOLO+SAM 到 YOLO-seg 的架构演化与 Jetson 极限部署全纪实

> 🌐 **[Read this in English](summary_en.md)**

本项目旨在面向农业水果（如苹果）采摘机器人，建立高精度、高实时性的视觉感知系统。从最初的级联模型到最终在老旧边缘设备上纯底层 API 的落地，整个项目经历了深度的优化与重构。

---

## 🚀 第一阶段：架构演进 (YOLOv8 + SAM ⟶ YOLOv8-seg)

### 1. 初始方案：YOLO 与 SAM 的级联组合
* **思路**：使用 YOLOv8 进行目标检测框选（Bounding Box），再将检测框作为 Prompt 传给大模型 SAM（如 MobileSAM / FastSAM）进行像素级多边形分割。
* **痛点**：速度极慢。即便在桌面级显卡上加入了 GPU Warmup 和批量 Prompt (Batched Inference) 优化，双模型的管线依然过于臃肿对于边缘设备来说极其致命。面对多目标场景时，SAM 的计算复杂度急剧上升。

### 2. 终极轻量化方案：YOLOv8-Seg 的端到端训练
* **思路**：直接训练单体 `YOLOv8n-seg` 模型，在一次前向传播中同时输出检测框与分割掩码 `O(1)` 时间复杂度。
* **破局 (伪标签自动标注)**：训练 Seg 模型通常需要极度昂贵的人工多边形画图成本。为此，我们开发了 **`auto_annotate.py`** 脚本。利用现有的真实检测框（Ground Truth BBoxes），在离线状态下调用 SAM 模型全自动生成 `AppleBBCH81` 数据集的 Polygon 掩码标注，实现了零人工成本的数据集升维。
* **成果**：在 `AppleBBCH81` 苹果数据集上重训后，精度 mAP50 达到 96.26%，同时完美做到了高帧率。

---

## 🛠️ 第二阶段：外围生态扩展
有了强大的核心模型后，我们开发了两套生态系统展示工具：
1. **MOT 产量统计 (`track_and_count.py`)**：结合 ByteTrack 算法与虚拟跨越线（Virtual Crossing Line），实现视频流内的苹果去重统计（追踪 ID 与计数）。解决依赖缺陷：安装了 `lapx` 库替补了丢失的 `lap` 模块。
2. **跨平台上位机 GUI (`gui_control_station.py`)**：抛弃了臃肿的 Web 或 PyQt 框架，仅依靠 Python 原生标准库 `Tkinter` + `OpenCV`，打造了极度轻量且跨平台的交互式工作站。实现了**实时滑动条调节置信度 (Conf)** 并动态刷新 Mask 边缘的效果。

---

## ☠️ 第三阶段：Jetson Nano B01 极限部署踩坑录

向 Jetson Nano B01 (2014年 Maxwell 架构，4GB共享内存) 的部署遭遇了极其恶劣的历史遗留软硬件限制，最终我们通过彻底脱离第三方高级库，用原生底层 API 杀出了一条血路。

### 坑点 1：老旧系统无法安装现代 Python 环境
* **表现**：Nano B01 被永远定格在了 JetPack 4.6.1 + Python 3.6.9 中，无法安装 `ultralytics` 包，直接卡死模型的端侧读取。
* **解决**：在上位机电脑上提前使用代码 (`export_jetson_tensorrt.py`) 将 pt 模型转为 `.onnx`。**关键设定：必须锁定 `opset=12` 且开启 `simplify=True`**，否则 Jetson 上自带的老旧 TensorRT 8.2 会报错算子不支持。

### 坑点 2：量化的绝对误区 (FP16 vs INT8)
* **表现**：盲目追求速度而在 Nano 上尝试 INT8 量化引擎。
* **解决**：**严禁使用 INT8，必须使用 `FP16`**。Jetson Nano 的 Maxwell 架构物理上根本没有张量核心 (Tensor Cores / DP4A)，不支持 INT8 硬件加速。强行跑 INT8 会回落为软解运算，不仅丧失 Mask 精度，甚至还不如跑 FP16。最终我们在板载终稿了 `.sh` 编译脚本：`trtexec --onnx=best.onnx --saveEngine=best.engine --fp16`。

### 坑点 3：底层 PyCUDA 的版本污染
* **表现**：在试图跑纯原生 TensorRT API 时，`import pycuda.driver` 报错找不到模块，系统里混乱共存了 Python 3.8 和 3.6。
* **解决**：强制清理 Python 3.8 的错误绑定。通过 `python3.6 -m pip install pycuda` 严格将 CUDA 底层内存操作库锚定在系统默认的 Python 3.6 环境。

### 坑点 4：CSI/USB 摄像头“绿化”现象 (Linux V4L2)
* **表现**：模型能推测出 25 FPS，但 `cv2.VideoCapture` 得到的画面却是一片纯绿色幽灵乱码。
* **解决**：
  * 若为 **CSI 硬件排线相机**：绿屏是因为读出的是未解码的 Bayer 原生像素。代码中写入 `GStreamer` 硬件解析管道 (`nvarguscamerasrc`) 唤醒英伟达 ISP 执行解马赛克运算。
  * 若为 **USB 免驱相机**：在 Linux 下默认走 YUYV 会因带宽断流导致矩阵全为 0 (YUV 0,0,0 = 绿色)。在代码中强力指定了 `cv2.CAP_V4L2` 以及 `MJPG` 或者 `YUYV` 参数。

### 坑点 5：徒手解包 TensorRT 矩阵引发的类型坍塌
* **表现**：使用纯 `tensorrt` Python API 推理后，拿到的是裸输出数组。为了重现 BBox NMS 取框逻辑，使用 Numpy 切片时触发了致命的 `TypeError: only size-1 arrays can be converted to Python scalars` 等数组异常，且有时候切出来的压根不是 Box。
* **解决**：
  1. **张量颠倒补丁**：发现部分转换后的 ONNX 引擎，Detection 输出和 Mask 输出顺序会互跳。所以用代码加入了检查机制：`det_output = raw_outputs[0] if len(raw_outputs[0].shape) == 3 else raw_outputs[1]` 自动寻回锚点张量；同时动态判断 `C` 通道和 `Anchors` 数的排列以实施矩阵 `.T` 转置。
  2. **原生类型净化**：OpenCV 绘图函数在老系统上极度脆弱。解包 `xc, yc, w, h` 时强行调用 `.item()` 将单元素 Numpy 标量“剥干净”，然后套上 Python 自带的 `float()` 和 `int()`，完美解决了坐标格式错位的问题。

### 坑点结论
最终配合缩小输入分辨率 (`imgsz=416`) 和解锁电源狂暴模式 (`nvpmodel -m 0`)，我们在一台无法安装 Ultralytics 的老古董板子上，使用纯底层 C++ 绑定流接口(`run_jetson_tensorrt.py`)，**极为硬核地压榨出了 25 帧以上/带 NMS 检测框画面的优质性能**。

---

## 📂 项目核心架构清单 (Current Layout)

- **数据集**
  - `data/AppleBBCH81/` : 经过重构且含有 Polygon Label 标签文本的数据集。
- **训练及上游阶段**
  - `auto_annotate.py` : 利用已知 Box，离线调用 SAM 全自动生成 Polygon txt 并落盘。
  - `train_yolo_seg.py` / `main_yolo_seg.py` : YOLOv8-seg 的训练构建与画图推理系统。
- **模型导出环节**
  - `export_jetson_tensorrt.py` : 在上位机环境用于锁定 `opset=12` 转码输出 `.onnx`。
- **边缘设备终极部署链 (Jetson 端)**
  - `build_engine.sh` : Nano 上的 `trtexec` 转编译批处理脚本。
  - `run_jetson_tensorrt.py` : 脱离 Ultralytics 框架，纯基于 PyCUDA 和 TensorRT 底层流的高速裸跑后处理+相机推理脚本。
- **扩展与展示态**
  - `gui_control_station.py` : 基于标准库构建的跨平台 GUI 界面。
  - `track_and_count.py` : 带有虚拟过线界盘和 ByteTrack 能力的视频流量清点逻辑。