# YOLO-AgriSeg-Jetson: The Architecture Evolution from YOLO+SAM to YOLO-seg and the Full Record of Extreme Jetson Deployment

This project aims to establish a high-precision, high-real-time visual perception system for agricultural fruit (such as apples) picking robots. From the initial cascaded model to the final deployment on outdated edge devices using pure low-level APIs, the entire project underwent profound optimization and refactoring.

---

## 🚀 Phase 1: Architecture Evolution (YOLOv8 + SAM ⟶ YOLOv8-seg)

### 1. Initial Solution: A Cascaded Combination of YOLO and SAM
* **Concept**: Using YOLOv8 for target detection (Bounding Box), and passing the detection boxes as Prompts to the large model SAM (like MobileSAM / FastSAM) for pixel-level polygon segmentation.
* **Pain Points**: Extremely slow speed. Even on desktop GPUs with GPU Warmup and Batched Inference optimizations, the dual-model pipeline remained exceedingly bloated, which proved fatal for edge devices. When faced with multi-target scenarios, SAM's computational complexity skyrocketed.

### 2. Ultimate Lightweight Solution: End-to-End Training of YOLOv8-Seg
* **Concept**: Directly training a monolithic `YOLOv8n-seg` model that outputs detection boxes and segmentation masks simultaneously in a single forward pass with `O(1)` time complexity.
* **Breakthrough (Pseudo-Label Auto-Annotation)**: Training an instance segmentation model typically requires exorbitantly expensive manual polygon drawing. To address this, we developed the **`auto_annotate_all.py`** script. By leveraging existing Ground Truth BBoxes, the SAM model is invoked fully offline to automatically generate Polygon mask annotations for the `AppleBBCH81` dataset, achieving zero-labor-cost dataset dimensionality upgrades.
* **Results**: After retraining on the `AppleBBCH81` apple dataset, mAP50 precision reached 96.26%, simultaneously maintaining perfect high frame rates.

---

## 🛠️ Phase 2: Peripheral Ecosystem Extension
With a robust core model established, we developed two ecosystem demonstration tools:
1. **MOT Yield Statistics (`track_and_count.py`)**: Combining the ByteTrack algorithm with Virtual Crossing Lines to realize de-duplicated apple counting (tracking ID and enumeration) in video streams. Dependency limitation solved: installed the `lapx` library as a substitute for the missing `lap` module.
2. **Cross-Platform Host GUI (`gui_control_station.py`)**: Abandoned bloated Web or PyQt frameworks, relying solely on Python native standard libraries `Tkinter` + `OpenCV` to build an extremely lightweight and cross-platform interactive workstation. Features **real-time slider adjustment of Confidence (Conf)** and a dynamic mask edge refresh effect.

---

## ☠️ Phase 3: Jetson Nano B01 Extreme Deployment Pitfall Guide

Deploying to the Jetson Nano B01 (2014 Maxwell architecture, 4GB shared memory) encountered severe historical hardware and software constraints. Ultimately, we broke through by completely detaching from third-party high-level libraries and utilizing pure native low-level APIs.

### Pitfall 1: Outdated systems cannot install modern Python environments
* **Symptoms**: The Nano B01 is permanently locked to JetPack 4.6.1 + Python 3.6.9, unable to install the `ultralytics` package, directly freezing model loading on the edge side.
* **Solution**: Ahead of time on the host PC, use script (`export_jetson_tensorrt.py`) to convert the pt model into `.onnx`. **Crucial Setting: You must lock `opset=12` and enable `simplify=True`**, otherwise the outdated TensorRT 8.2 native to the Jetson will report unsupported operations.

### Pitfall 2: The Absolute Fallacy of Quantization (FP16 vs INT8)
* **Symptoms**: Blindly pursuing speed by attempting INT8 quantization engines on the Nano.
* **Solution**: **Strictly forbid INT8, must use `FP16`**. The Jetson Nano's Maxwell architecture physically lacks Tensor Cores (DP4A), offering zero INT8 hardware acceleration. Forcing INT8 relies on software emulation, resulting in both the loss of Mask accuracy and slower speeds than FP16. We finalized a `.sh` compilation script on board: `trtexec --onnx=best.onnx --saveEngine=best.engine --fp16`.

### Pitfall 3: Low-level PyCUDA Version Contamination
* **Symptoms**: When attempting to run the pure native TensorRT API, `import pycuda.driver` throws a module not found error. Python 3.8 and 3.6 chaotically co-existed in the system.
* **Solution**: Forcibly clear erroneous bindings from Python 3.8. Use `python3.6 -m pip install pycuda` to rigidly anchor CUDA's low-level memory operation library to the system's default Python 3.6 environment.

### Pitfall 4: CSI/USB Camera "Green Screen" Phenomenon (Linux V4L2)
* **Symptoms**: The model could infer at 25 FPS, but `cv2.VideoCapture` yielded a pure green ghost matrix.
* **Solution**:
  * For **CSI Hardware Ribbon Cameras**: The green screen was because raw Bayer pixels were read undefeated. Code was rewritten to integrate `GStreamer` hardware parsing pipelines (`nvarguscamerasrc`) to awake NVIDIA's ISP for demosaicing computations.
  * For **Driverless USB Cameras**: Under Linux, defaulting to YUYV can cause matrix zero-outs due to bandwidth stuttering (YUV 0,0,0 = green). Strong directives for `cv2.CAP_V4L2` and `MJPG` or `YUYV` parameters were forced into the code.

### Pitfall 5: Type Collapse from Manually Unpacking TensorRT Matrices
* **Symptoms**: Using the pure `tensorrt` Python API yields bare output arrays. When using Numpy slicing to recreate the BBox NMS logic, a fatal `TypeError: only size-1 arrays can be converted to Python scalars` occurred, and sometimes the sliced arrays were not even Boxes.
* **Solution**:
  1. **Tensor Inversion Patch**: Discovered that some ONNX engines scramble Detection and Mask output sequences. A checking mechanism was added: `det_output = raw_outputs[0] if len(raw_outputs[0].shape) == 3 else raw_outputs[1]` to automatically find the anchor tensor; combined with dynamic `C` dimension checks to apply `.T` transpose.
  2. **Native Type Purification**: OpenCV drawing functions are extremely fragile on older systems. Unpacked `xc, yc, w, h` using `.item()` to "peel bare" the single-element Numpy scalar, wrapping it with native python `float()` and `int()`, flawlessly resolving format mismatches.

### Pitfalls Conclusion
Finally, coupled with reducing input resolution (`imgsz=416` or `imgsz=320`) and unlocking the max power mode (`nvpmodel -m 0`), we pushed an antiquated board—incapable of installing Ultralytics—to its absolute limits using a pure C++ binding stream interface (`run_jetson_tensorrt.py`), **solidly squeezing out 25+ FPS with NMS-drawn detection arrays**.

---

## 📂 Core Architecture Inventory (Current Layout)

- **Dataset**
  - `data/AppleBBCH81/` : The reconstructed dataset containing Polygon Label text.
- **Training & Upstream Stages**
  - `auto_annotate_all.py` : Utilizes known Boxes offline to call SAM, generating Polygon txts directly to disk.
  - `train_yolo_seg.py` / `main_yolo_seg.py` : YOLOv8-seg training constructions and plotting inference systems.
- **Model Export Stage**
  - `export_jetson_tensorrt.py` : Used in the host environment to lock `opset=12` for `.onnx` transcoding.
- **Extreme Edge Device Deployment Chain (Jetson Side)**
  - `build_engine.sh` : The `trtexec` batch compilation script on the Nano.
  - `run_jetson_tensorrt.py` : Decoupled from Ultralytics, a pure PyCUDA and TensorRT low-level streaming high-speed bare-bones inference script.
- **Extensions and Demonstrations**
  - `gui_control_station.py` : A cross-platform GUI interface built on standard libraries.
  - `track_and_count.py` : Video flow counting logic armed with virtual crossing bounds and ByteTrack capabilities.