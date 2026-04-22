"""
利用零样本分割模型 SAM，根据已有目标框(Bounding Box)全自动将整个数据集离线处理成多边形掩码(Polygon Mask)标签文本，从而为训练 YOLOv8-seg 模型提供强监督掩码真值，省去了大量人力标注。
"""
import os
import cv2
import numpy as np
import torch
import shutil
from tqdm import tqdm
from ultralytics import SAM

def xcycwh2xyxy(x, y, w, h, img_w, img_h):
    """将中心点和宽高转换为绝对坐标 xyxy"""
    x_center = x * img_w
    y_center = y * img_h
    width = w * img_w
    height = h * img_h
    return [x_center - width / 2, y_center - height / 2, x_center + width / 2, y_center + height / 2]

def convert_dataset(base_dir):
    print("[INFO] 开始加载 MobileSAM 并构建自动标注器...")
    sam_model = SAM("mobile_sam.pt")
    
    labels_dir = os.path.join(base_dir, 'labels')
    labels_backup = os.path.join(base_dir, 'labels_bbox_backup')
    
    if not os.path.exists(labels_backup):
        print(f"[INFO] 为了安全起见，备份原始 Bbox 标签到 {labels_backup} ...")
        shutil.copytree(labels_dir, labels_backup)
        
    for split in ['train', 'val']:
        img_split_dir = os.path.join(base_dir, 'images', split)
        label_split_dir = os.path.join(base_dir, 'labels', split)
        
        if not os.path.exists(img_split_dir):
            continue
            
        img_files = [f for f in os.listdir(img_split_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        print(f"\n[INFO] 正在处理 {split} 集, 共 {len(img_files)} 张图像...")
        
        for img_name in tqdm(img_files):
            img_path = os.path.join(img_split_dir, img_name)
            base_name = os.path.splitext(img_name)[0]
            label_path = os.path.join(label_split_dir, base_name + '.txt')
            
            if not os.path.exists(label_path):
                continue
                
            with open(label_path, 'r') as f:
                lines = f.readlines()
            if not lines:
                continue
                
            img = cv2.imread(img_path)
            if img is None: continue
            h_img, w_img = img.shape[:2]
            
            classes = []
            boxes = []
            
            # 解析本图中的所有 Box
            for line in lines:
                parts = line.strip().split()
                # 只处理尚未转换为 polygon 的标准 bbox
                if len(parts) == 5:
                    cls_id = parts[0]
                    cx, cy, w, h = map(float, parts[1:5])
                    abs_box = xcycwh2xyxy(cx, cy, w, h, w_img, h_img)
                    classes.append(cls_id)
                    boxes.append(abs_box)
                elif len(parts) > 5:
                    # 如果已经是多边形，就跳着处理或者直接不管（为了兼容中断恢复）
                    pass
            
            if not boxes:
                continue
                
            # SAM 批量转化
            results = sam_model.predict(img, bboxes=boxes, verbose=False)
            polygons_lines = []
            
            if len(results) > 0 and getattr(results[0], 'masks', None) is not None:
                masks = results[0].masks.data.cpu().numpy().astype(np.uint8)
                
                for i, mask in enumerate(masks):
                    mask_resized = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
                    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if len(contours) == 0:
                        # 如果极端情况下没有找到 Mask(非常小)，平替为外切矩形多边形
                        x1, y1, x2, y2 = boxes[i]
                        polygons_lines.append(f"{classes[i]} {x1/w_img:.6f} {y1/h_img:.6f} {x2/w_img:.6f} {y1/h_img:.6f} {x2/w_img:.6f} {y2/h_img:.6f} {x1/w_img:.6f} {y2/h_img:.6f}")
                        continue
                        
                    largest_contour = max(contours, key=cv2.contourArea)
                    
                    # 轮廓点近似，减少由于锯齿造成的点数过多
                    epsilon = 0.002 * cv2.arcLength(largest_contour, True)
                    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                    
                    poly_str = []
                    for pt in approx:
                        x, y = pt[0]
                        poly_str.extend([f"{x/w_img:.6f}", f"{y/h_img:.6f}"])
                        
                    if len(poly_str) >= 6: # 至少三个坐标点 (6个数值)
                        polygons_lines.append(f"{classes[i]} " + " ".join(poly_str))
                    else:
                        x1, y1, x2, y2 = boxes[i]
                        polygons_lines.append(f"{classes[i]} {x1/w_img:.6f} {y1/h_img:.6f} {x2/w_img:.6f} {y1/h_img:.6f} {x2/w_img:.6f} {y2/h_img:.6f} {x1/w_img:.6f} {y2/h_img:.6f}")

            # 覆盖原文件
            with open(label_path, 'w') as f:
                f.write("\n".join(polygons_lines) + "\n")
                
    print("\n✅ 所有图像转换已完毕！现在的 data/AppleBBCH81 已经是一个标准的 Instance Segmentation (实例分割) 数据集了！")

if __name__ == '__main__':
    base_dir = "./data/AppleBBCH81"
    convert_dataset(base_dir)