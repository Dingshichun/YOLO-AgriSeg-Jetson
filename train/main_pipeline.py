"""
早期的 YOLO (提供 Box) 与 SAM (提供 Mask) 的级联推理流水线版本，展示了基于 Prompt 驱动的视觉分割测试方案。
"""
import os
import cv2
import numpy as np
import torch
import time
from ultralytics import YOLO

def main():
    """农业机器人视觉流水线"""
    image_path = "data/AppleBBCH81/images/val/DSC_1042_17kv1r16k_0.jpg"
    yolo_weight = "models/weights/best.engine"
    
    print("[1/3] 初始化模型...")
    yolo_model = YOLO(yolo_weight, task="detect")
    
    from ultralytics import SAM
    sam_model = SAM("mobile_sam.pt")
    
    # === 1: 预热 GPU 与引擎 ===
    print("[INFO] Warmup GPU context... (消除首次推理建立图的耗时)")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    yolo_model.predict(dummy, verbose=False)
    sam_model.predict(dummy, bboxes=[0,0,10,10], verbose=False)
    torch.cuda.synchronize() # 强制同步以确保预热彻底结束
    # ======================================

    print("[3/3] 运行串联流水线...")
    image = cv2.imread(image_path)
    
    # 避免受到模型加载时间的干扰，真正计时从这里开始
    start_time = time.time()
    
    # 第一步：YOLO 推理
    yolo_results = yolo_model(image, verbose=False)[0]
    boxes = yolo_results.boxes.xyxy.cpu().numpy()
    torch.cuda.synchronize()
    t1 = time.time()
    
    print(f"[INFO] YOLO 纯推理耗时: {(t1-start_time)*1000:.2f} ms")
    
    if len(boxes) == 0: return

    overlay_image = image.copy()
    
    # 第二步：SAM 推理 (可以进行批处理优化)
    sam_t0 = time.time()
    
    # === 核心优化点 2: Batched Prompt 批处理优化 ===
    # 直接把多个 Box 以张量丢入 SAM 一次性前向传播
    sam_results = sam_model.predict(image, bboxes=boxes, verbose=False)
    torch.cuda.synchronize()
    sam_t1 = time.time()
    
    print(f"[INFO] SAM 并发预估所有框({len(boxes)}个)耗时: {(sam_t1-sam_t0)*1000:.2f} ms")
    
    # 后处理(OpenCV overlay不计算在纯模型推理耗时内，但计算在总延迟内)
    post_t0 = time.time()
    masks = []
    if len(sam_results) > 0 and getattr(sam_results[0], 'masks', None) is not None:
        # 获取合并后的 masks
        masks = sam_results[0].masks.data.cpu().numpy().astype(np.uint8)
        
        for idx in range(len(masks)):
            mask = cv2.resize(masks[idx], (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            color = np.random.randint(0, 255, (3,), dtype=np.uint8).tolist()
            roi = mask > 0
            overlay_image[roi] = overlay_image[roi] * 0.5 + np.array(color) * 0.5
            box = boxes[idx]
            cv2.rectangle(overlay_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            
    post_t1 = time.time()
    
    end_time = time.time()
    print(f"=====================================")
    print(f"[Profiling] YOLO 延迟 : {(t1-start_time)*1000:6.2f} ms")
    print(f"[Profiling] SAM  延迟 : {(sam_t1-sam_t0)*1000:6.2f} ms")
    print(f"[Profiling] 后处理延迟: {(post_t1-post_t0)*1000:6.2f} ms")
    print(f"[Profiling] 端到端FPS : {1.0 / (end_time-start_time):6.2f} 帧/秒")
    print(f"=====================================")

    os.makedirs("output", exist_ok=True)
    cv2.imwrite("output/masked_result.jpg", overlay_image)

if __name__ == "__main__":
    main()
