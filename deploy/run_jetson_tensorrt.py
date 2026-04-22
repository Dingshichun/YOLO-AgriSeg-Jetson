"""
资源受限环境 ( Jetson Nano B01) 的核心推理部署代码，在 jetson 上运行。
彻底剥离了沉重的 Ultralytics 库依赖，完全采用裸 TensorRT/PyCUDA C++-Bindings，
配合手工张量降维矩阵拆解、原生的 非极大值抑制(Non-Maximum suppression, NMS) 防交叠等，实现最高帧率。
"""
import cv2
import numpy as np
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# 初始化 TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TRT_YOLO_Seg:
    def __init__(self, engine_path, input_shape=(640, 640)):
        print(f"正在通过原生 TensorRT API 加载引擎: {engine_path}")
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_shape = input_shape
        
        # 分配显存
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        for binding in self.engine:
            shape = tuple(self.engine.get_binding_shape(binding))
            size = trt.volume(self.engine.get_binding_shape(binding)) * \
                   self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # 分配主机端与设备端(GPU)内存
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
                
    def infer(self, img_bgr):
        # 1. Preprocess
        h, w = img_bgr.shape[:2]
        img_resized = cv2.resize(img_bgr, self.input_shape)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_chw = np.transpose(img_rgb, (2, 0, 1)).astype(np.float32) / 255.0
        img_blob = np.expand_dict(img_chw, axis=0) if not 'np.expand_dims' else np.expand_dims(img_chw, axis=0)
        
        np.copyto(self.inputs[0]['host'], img_blob.ravel())
        
        # 2. 从主机(CPU)拷贝到设备(GPU)
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # 3. 异步推理
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # 4. 从设备(GPU)拷贝回主机(CPU)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        self.stream.synchronize()
        
        # 返回原始输出阵列，恢复正确的维度形状
        return [out['host'].reshape(out['shape']) for out in self.outputs]

def main():
    engine_path = "imgsz416_best.engine"  # 建议把上一步在Nano上生成的engine放到同目录
    imgsz = (416, 416)  # 这个尺寸必须和你生成engine时的输入尺寸一致，否则推理结果会完全错乱
    trt_model = TRT_YOLO_Seg(engine_path, input_shape=imgsz)
    print("模型加载与显存分配结束！")

    # ========== 选择图片输入源 ==========
    # 针对 Jetson Nano 的特殊硬件架构，有可能持续产生“绿色乱码”或纯绿屏的核心原因通常是：
    # ① 使用的是【CSI 排线摄像头】(如带有排线的 IMX219/树莓派相机)，
    #    此时系统拿到的是原始 Bayer 像素阵列，如果不经过 GStreamer 调用硬件 ISP 进行去马赛克(Demosaic)，图像将被完全误解为全绿色！
    # ② 使用的是【USB 摄像头】，且该摄像头不支持 MJPG 被强行解码失败。
    
    # 下面是两种不同摄像头的加载方式。我使用的是 CSI 排线相机，所以默认开启了 GStreamer 加速通道：

    # ---- 选项 A: CSI 排线硬件相机 (默认开启) ----
    gstreamer_pipeline = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12, framerate=30/1 ! "
        "nvvidconv flip-method=0 ! "
        "video/x-raw, width=640, height=640, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
    )
    cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)

    # ---- 选项 B: 传统 USB 摄像头 
    # cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV')) # YUYV是免驱摄像头最原始也最通用的格式，可以避免 MJPG 解码失败导致的绿屏问题
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    
    if not cap.isOpened():
        print("错误：无法打开摄像头。")
        return

    print("开始纯 TensorRT 推理，按 'q' 键退出...")
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        t1 = time.time()

        # 执行极限推理
        raw_outputs = trt_model.infer(frame)
        
        # ========== 开始手动后处理 (NMS + 绘制矩形框) ==========
        # TensorRT 导出的 engine 中，检测头和 Mask 头输出的顺序可能会颠倒
        # 边界框输出形状通常为 3 维 (1, C, Anchors)，而 Mask 的原型输出为 4 维 (1, 32, mask_h, mask_w)
        det_output = raw_outputs[0] if len(raw_outputs[0].shape) == 3 else raw_outputs[1]
        
        output0 = det_output[0] # 提取第 0 个 Batch，降维成 (C, Anchors) 或 (Anchors, C)
        
        # 动态判断哪维是特征通道(C)，如果 C 排在前面，则转置
        if output0.shape[0] < output0.shape[1]:
            output0 = output0.T         # 转置为 (Anchors, C) 防止缓存不连续
        
        num_mask_coeffs = 32
        dimensions = output0.shape[1]
        num_classes = dimensions - 4 - num_mask_coeffs
        
        # 提取各个部分
        bboxes = output0[:, 0:4]                        # 格式 [xc, yc, w, h]
        class_scores = output0[:, 4:4+num_classes]      # 各类置信度
        
        # 寻找每个预测框中最大置信度的类别，及其对应的分值
        max_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)
        
        # 设定置信度阈值 (Conf_threshold)
        conf_thresh = 0.25
        valid_indices = np.where(max_scores > conf_thresh)[0]
        
        filtered_boxes = bboxes[valid_indices]
        filtered_scores = max_scores[valid_indices]
        filtered_class_ids = class_ids[valid_indices]
        
        # 针对图片拉伸还原比例 (由于此前推理是直接暴力 resize 到 416 或 640)
        h_orig, w_orig = frame.shape[:2]
        x_scale = w_orig / trt_model.input_shape[0]
        y_scale = h_orig / trt_model.input_shape[1]
        
        cv2_boxes = []
        for box in filtered_boxes:
            xc, yc, w_box, h_box = box
            # 转换为原图像素坐标系，用 .item() 强行将其从 numpy 标量里拔出来剥离干净
            xc = float(xc.item() * x_scale)
            w_box = float(w_box.item() * x_scale)
            yc = float(yc.item() * y_scale)
            h_box = float(h_box.item() * y_scale)
            
            x_min = xc - (w_box / 2.0)
            y_min = yc - (h_box / 2.0)
            cv2_boxes.append([int(x_min), int(y_min), int(w_box), int(h_box)])
            
        # 利用 OpenCV 提供的高性能 NMS (非极大值抑制)，滤除重叠检测框
        iou_thresh = 0.45
        nms_indices = cv2.dnn.NMSBoxes(cv2_boxes, filtered_scores.tolist(), conf_thresh, iou_thresh)
        
        if len(nms_indices) > 0:
            for i in nms_indices.flatten():
                idx = int(i)
                x, y, w_b, h_b = cv2_boxes[idx]
                score = float(filtered_scores[idx])
                
                # 画框
                cv2.rectangle(frame, (x, y), (x + w_b, y + h_b), (0, 255, 0), 2)
                # 画置信度文本
                text = f"Apple: {score:.2f}"
                cv2.putText(frame, text, (x, max(10, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # =============================================================

        t2 = time.time()
        fps = 1.0 / (t2 - t1)
        
        cv2.putText(frame, f"NANO TRT API FPS: {fps:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

        cv2.imshow("Pure TensorRT Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
