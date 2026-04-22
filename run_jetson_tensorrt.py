"""
专为资源受限环境 (如 Jetson Nano B01) 打造的核心推理部署代码。
彻底剥离了沉重的 Ultralytics 库依赖，完全采用裸 TensorRT/PyCUDA C++-Bindings，配合手工张量降维矩阵拆解、原生的 NMS 防交叠等，实现最高帧率。
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
        # 1. 预处理 (Preprocess)
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
    engine_path = "best.engine"  # 请确保把上一步在Nano上生成的engine放到同目录
    trt_model = TRT_YOLO_Seg(engine_path, input_shape=(640, 640))
    print("模型加载与显存分配结束！")

    # ========== 选择输入源 ==========
    # 针对Jetson Nano的特殊硬件架构，持续产生“绿色乱码”或纯绿屏的核心原因通常是：
    # ① 您使用的是【CSI 排线摄像头】(如带有排线的IMX219/树莓派相机)，此时系统拿到的是原始Bayer像素阵列，如果不经过GStreamer调用硬件ISP进行去马赛克(Demosaic)，图像将被完全误解为全绿色！
    # ② 您使用的是【USB摄像头】，且该摄像头不支持 MJPG 被强行解码失败。
    
    # 💥 【修复方案】下面为您提供了两种不同摄像头的加载方式。强烈推测您在使用CSI相机，所以默认开启了GStreamer加速通道：

    # ---- 选项 A: CSI 排线硬件相机 (默认开启) ----
    gstreamer_pipeline = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12, framerate=30/1 ! "
        "nvvidconv flip-method=0 ! "
        "video/x-raw, width=640, height=480, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
    )
    cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)

    # ---- 选项 B: 传统 USB 摄像头 (如使用USB，请注释上方A选项，并取消下方注释) ----
    # cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV')) # YUYV是免驱摄像头最原始也最通用的格式，防止解霸绿屏
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
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
        
        output0 = det_output[0] # 提取第0个Batch，降维成 (C, Anchors) 或 (Anchors, C)
        
        # 动态判断哪维是特征通道(C)，如果C排在前面，则转置
        if output0.shape[0] < output0.shape[1]:
            output0 = output0.T         # 转置为 (Anchors, C) 防止缓存不连续
        
        num_mask_coeffs = 32
        dimensions = output0.shape[1]
        num_classes = dimensions - 4 - num_mask_coeffs
        
        # 提取各个部分
        bboxes = output0[:, 0:4]                        # 格式 [xc, yc, w, h]
        class_scores = output0[:, 4:4+num_classes]      # 各类置信度
        mask_coeffs = output0[:, 4+num_classes:]        # Mask 掩码系数 (32 维)
        
        # 寻找每个预测框中最大置信度的类别，及其对应的分值
        max_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)
        
        # 设定置信度阈值 (Conf_threshold)
        conf_thresh = 0.25
        valid_indices = np.where(max_scores > conf_thresh)[0]
        
        filtered_boxes = bboxes[valid_indices]
        filtered_scores = max_scores[valid_indices]
        filtered_class_ids = class_ids[valid_indices]
        filtered_mask_coeffs = mask_coeffs[valid_indices]
        
        # 针对图片拉伸还原比例 (由于此前推理是暴力 resize到 416 或 640)
        h_orig, w_orig = frame.shape[:2]
        x_scale = w_orig / trt_model.input_shape[0]
        y_scale = h_orig / trt_model.input_shape[1]
        
        cv2_boxes = []
        for box in filtered_boxes:
            xc, yc, w_box, h_box = box
            # 转换为原图像素坐标系，这回用 .item() 强行将其从 numpy 标量里拔出来剥离干净
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
                
                # ========================================================
                # [进阶功能：多边形 Mask 像素级掩码渲染 (控制机械臂抓取点位)]
                # ⚠️ 警告：在 Jetson Nano 的 CPU 跑 NumPy 稠密矩阵乘法会导致帧率暴降。
                # 若需要将苹果真实形状喂给机械臂避障使用，去除下面单引号注释即可：
                '''
                # 1. 提取当前对应苹果的 32 维掩码系数
                m_c = filtered_mask_coeffs[idx]
                
                # 2. 获取原版模型 Mask 原型 (结构通常为: 1, 32, Mh, Mw)
                proto = raw_outputs[0] if len(raw_outputs[0].shape) == 4 else raw_outputs[1]
                proto = proto[0] # (32, Mh, Mw)
                c_dim, mh, mw = proto.shape
                
                # 3. 矩阵乘法：系数(32,) @ 原型张量拉平展开(32, Mh*Mw)
                mask = np.dot(m_c, proto.reshape(c_dim, -1)).reshape(mh, mw)
                
                # 4. Sigmoid 激活函数将其映射至 0~1 的置信度
                mask = 1.0 / (1.0 + np.exp(-mask))
                
                # 5. 缩放上采样还原到此时帧的真正像素分辨率
                mask = cv2.resize(mask, (w_orig, h_orig))
                
                # 6. 二值化 (大于 0.5 的像素点判定为果实区域)
                mask_bin = (mask > 0.5).astype(np.uint8)
                
                # 7. 画面抑制：裁去不处于矩形 Bbox 里的杂乱无章游离星点
                crop_mask = np.zeros_like(mask_bin)
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(w_orig, x + w_b), min(h_orig, y + h_b)
                crop_mask[y1:y2, x1:x2] = mask_bin[y1:y2, x1:x2]
                
                # 8. RGB 色彩叠加渲染成琥珀色蒙版
                colored_mask = np.zeros_like(frame)
                colored_mask[crop_mask == 1] = [0, 165, 255]
                frame = cv2.addWeighted(frame, 1.0, colored_mask, 0.4, 0)
                '''
                # ========================================================
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
