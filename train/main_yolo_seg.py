"""
单体端到端 YOLOv8-seg 高速推理主程序。
抛弃了冗余的 SAM 级联结构，实现 O(1) 复杂度的超快实时识别及多边形原生地生成，属于性能成熟期架构。
"""
import os
import cv2
import numpy as np
import torch
import time
from ultralytics import YOLO

def main():
    """农业机器人视觉流水线 (端到端高速单模型方案: YOLOv8-seg)"""
    
    # 需要先使用 train_yolo_seg.py 训练完了模型，路径在 runs/segment/train... 下
    # 当导出为 tensorrt 之后，也可以将这里换成 .engine
    yolo_weight = "runs/segment/train/weights/best.pt" 
    image_path = "data/AppleBBCH81/images/val/DSC_1042_17kv1r16k_0.jpg"
    
    print("[1/2] 初始化 YOLOv8-Seg 模型...")
    if not os.path.exists(yolo_weight):
        print(f"[WARN] 权重文件 {yolo_weight} 不存在。请确认训练是否完成，如果是初次尝试，可以先用 'yolov8n-seg.pt' 看看效果。")
        # 可以临时改回 "yolov8n-seg.pt" 测试代码
    
    try:
        model = YOLO(yolo_weight, task="segment")
    except Exception as e:
        print(f"[ERROR] 模型加载失败: {e}")
        return
        
    # === 预热 GPU 与引擎 ===
    print("[INFO] Warmup GPU context... (消除首次推理建立计算图的耗时)")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(dummy, verbose=False)
    torch.cuda.synchronize() # 强制同步
    # ================================

    print("[2/2] 运行端到端单模型预测...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] 读取图像失败: {image_path}")
        return
        
    start_time = time.time()
    
    # 纯推理阶段，YOLOv8-seg 这一次前向传播就会同时干完 YOLO + SAM 过去各自干的活！
    # retina_masks=True 可以让模型输出与原图分辨率一致的高清 Mask，而非下采样后的低清 Mask
    results = model.predict(image, retina_masks=True, verbose=False)[0]
    torch.cuda.synchronize()
    infer_time = time.time()
    
    if results.boxes is None or len(results.boxes) == 0:
        print("[INFO] 未检测到任何目标。")
        return
        
    boxes = results.boxes.xyxy.cpu().numpy()
    print(f"[INFO] 成功一站式预估出 {len(boxes)} 个目标的 BBox 与 Mask。")
    
    overlay_image = image.copy()
    
    # 后处理(OpenCV overlay不计算在纯模型推理耗时内)
    post_t0 = time.time()
    
    if results.masks is not None:
        masks = results.masks.data.cpu().numpy().astype(np.uint8)
        
        for idx in range(len(boxes)):
            mask = masks[idx]
            
            # 因为使用了 retina_masks=True，尺寸已经对齐，但为了安全我们依然统一进行 Resize 核对
            if mask.shape[0] != image.shape[0] or mask.shape[1] != image.shape[1]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                
            color = np.random.randint(0, 255, (3,), dtype=np.uint8).tolist()
            
            # 绘制半透明掩码
            roi = mask > 0
            overlay_image[roi] = overlay_image[roi] * 0.5 + np.array(color) * 0.5
            
            # 画上 Bbox 矩形框
            box = boxes[idx]
            cv2.rectangle(overlay_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
    else:
        print("[WARN] 模型没有输出任何 Mask！")
        
    post_t1 = time.time()
    
    print(f"==========================================")
    print(f"[Profiling] 单模型极限推理: {(infer_time-start_time)*1000:6.2f} ms")
    print(f"[Profiling] 渲染及后处理耗时: {(post_t1-post_t0)*1000:6.2f} ms")
    print(f"[Profiling] 终端系统综合 FPS: {1.0 / (post_t1-start_time):6.2f} 帧/秒")
    print(f"==========================================")

    os.makedirs("output", exist_ok=True)
    out_path = "output/yolov8_seg_result.jpg"
    cv2.imwrite(out_path, overlay_image)
    print(f"[INFO] 结果已保存至: {out_path}")

if __name__ == "__main__":
    main()