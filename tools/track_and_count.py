"""
使用 ByteTrack 进行 多目标追踪 算法的叠加试验。
实现针对画面通过给定判定线界的前后虚拟计数功能，常用于农业流水线或摄像头行走的苹果产量估算。
"""
import cv2
import os
import numpy as np
from ultralytics import YOLO

def process_tracking(source_path, model_path):
    if not os.path.exists(source_path):
        print(f"[ERROR] 找不到输入源: {source_path}")
        return
        
    print(f"[INFO] 正在加载模型: {model_path}")
    try:
        model = YOLO(model_path, task="segment")
    except:
        print("[WARN] 本地自训模型加载失败，回退到预训练 yolov8n-seg.pt")
        model = YOLO("yolov8n-seg.pt", task="segment")

    img = cv2.imread(source_path)
    if img is None:
        print("[ERROR] 图像加载失败")
        return
        
    h, w = img.shape[:2]
    
    # 定义一条垂直的虚拟越线判定区域（画在画面正中间）
    line_x = int(w / 2)
    
    print("[INFO] 开始进行带 Tracking ID 的目标跟踪与实例分割...")
    # persist=True 保留历史轨迹记忆，基于 ByteTrack 算法
    # 由于我只有图，跟踪器主要是为画面中出现的首批目标分配「初始追踪专属 ID」
    results = model.track(img, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
    
    overlay = img.copy()
    
    # 1. 绘制越线判定区 (黄色粗线)
    cv2.line(overlay, (line_x, 0), (line_x, h), (0, 255, 255), 4)
    cv2.putText(overlay, "-- Virtual Crossing Line --", (line_x + 15, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    
    current_count = 0
    if results.boxes is not None and results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        # 获取跟踪器分配的唯一 ID 列表
        track_ids = results.boxes.id.int().cpu().tolist()
        
        # 获取预测出的像素级 Mask
        masks = None
        if results.masks is not None:
            masks = results.masks.data.cpu().numpy().astype(np.uint8)
            
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            track_id = track_ids[i]
            # 计算苹果在这一帧里的几何中心坐标
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            
            # 为每个独立的 ID 分配一种随机颜色作为区分
            np.random.seed(track_id * 100) # 根据 ID 固定随机数种子，保证连续帧内同一个目标的颜色不会疯狂闪烁
            color = [int(c) for c in np.random.randint(0, 255, size=(3,))]
            
            # 2. 绘制半透明 Mask
            if masks is not None:
                mask = cv2.resize(masks[i], (w, h), interpolation=cv2.INTER_NEAREST)
                roi = mask > 0
                overlay[roi] = overlay[roi] * 0.6 + np.array(color) * 0.4
            
            # 3. 绘制 BBox、当前帧中心点 (重要：越线的物理依据)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
            cv2.circle(overlay, (cx, cy), 8, (0, 0, 255), -1) # 中心点用红色圆点标出
            
            # 4. 绘制悬浮的 Tracking ID 
            label = f"ID: #{track_id}"
            cv2.putText(overlay, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            
            # 【提示】：这只是单张静态图的模拟测试！
            # 真实视频的产量评估逻辑是判断上一帧这颗苹果在黄线左侧，当前帧到了右侧：
            # if previous_cx[track_id] < line_x and current_cx >= line_x:
            #     Total_Yield += 1
            # 在这里仅为画面中所有成功分配 ID 的果子计入模拟总数
            current_count += 1
            
    else:
        print("[WARN] 未检测到任何带有跟踪 ID 的苹果目标。")
        
    # 5. 绘制左上角的 【总产量评估面板】
    # 黑色半透明背景面板
    panel_w, panel_h = 420, 100
    sub_overlay = overlay.copy()
    cv2.rectangle(sub_overlay, (20, 20), (20 + panel_w, 20 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(sub_overlay, 0.6, overlay, 0.4, 0, overlay)
    
    # 绿色数字显示
    cv2.putText(overlay, f"Yield Count: {current_count} (Mock)", (40, 85), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 4)
                
    os.makedirs("output", exist_ok=True)
    out_path = "output/track_count_test.jpg"
    cv2.imwrite(out_path, overlay)
    print(f"\n✅ 静态图片 Tracking 模型 & Count 逻辑可视化测试成功！")
    print(f"由于是一张静态图片，画面中分配的轨迹 ID 是第一帧初始状态。")
    print(f"结果图片已保存至: {out_path}")

if __name__ == "__main__":
    # 使用一张验证集苹果图片模拟果树漫游过程中的第一帧
    img_path = "data/AppleBBCH81/images/val/DSC_1042_17kv1r16k_0.jpg"
    model_path = "runs/segment/train/weights/best.pt"
    process_tracking(img_path, model_path)