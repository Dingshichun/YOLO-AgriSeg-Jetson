"""
基于 Python 原生 Tkinter 构建的极度轻量化的上位机交互式控制工作站。
支持动态滑动条调节推理置信度、选择图像/视频/摄像头等多种数据源进行纯视觉验证演示呈现。
"""
import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO

class AgriRobotGUI:
    def __init__(self, root, title):
        self.root = root
        self.root.title(title)
        self.root.geometry("1000x700")
        
        # 初始化感知大模型
        print("正在加载感知模型...")
        try:
            self.model = YOLO("runs/segment/train/weights/best.pt", task="segment")
            print("加载本地自训 YOLO-seg 成功。")
        except:
            print("找不到自训模型，使用官方默认轻量模型 yolov8n-seg.pt")
            self.model = YOLO("yolov8n-seg.pt", task="segment")
        
        self.cap = None             # OpenCV VideoCapture 对象
        self.is_playing = False     # 当前是否在播放视频流
        self.current_frame = None   # 暂存的原图帧
        self.after_id = None        # Tkinter 定时器 ID
        
        self._build_ui()
        
    def _build_ui(self):
        """构建控制台用户界面"""
        # ======================= 控制面板栏 =======================
        control_frame = tk.Frame(self.root, bg="#2C3E50", pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # 按钮样式
        btn_style = {"bg": "#1ABC9C", "fg": "white", "font": ("Microsoft YaHei", 11, "bold"), "relief": tk.FLAT, "padx": 10}
        
        self.btn_img = tk.Button(control_frame, text="导入单图", command=self.load_image, **btn_style)
        self.btn_img.pack(side=tk.LEFT, padx=10)
        
        self.btn_vid = tk.Button(control_frame, text="导入视频", command=self.load_video, **btn_style)
        self.btn_vid.pack(side=tk.LEFT, padx=10)
        
        self.btn_cam = tk.Button(control_frame, text="开启本地相机", command=self.open_camera, **btn_style)
        self.btn_cam.pack(side=tk.LEFT, padx=10)
        
        self.btn_stop = tk.Button(control_frame, text="停止/清空", command=self.stop_stream, bg="#E74C3C", fg="white", font=("Microsoft YaHei", 11, "bold"), relief=tk.FLAT, padx=10)
        self.btn_stop.pack(side=tk.LEFT, padx=10)
        
        # 动态置信度调节滑动条
        self.lbl_conf = tk.Label(control_frame, text="置信度阈值 (Conf):", bg="#2C3E50", fg="white", font=("Microsoft YaHei", 10))
        self.lbl_conf.pack(side=tk.LEFT, padx=(20, 5))
        
        self.conf_slider = tk.Scale(control_frame, from_=0.01, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, bg="#2C3E50", fg="white", highlightthickness=0, command=self.on_conf_change)
        self.conf_slider.set(0.25) # 默认 25% 置信度
        self.conf_slider.pack(side=tk.LEFT)
        
        # OSD 信息标签
        self.status_bar = tk.Label(self.root, text="系统已就绪，请选择输入源。", bd=1, relief=tk.SUNKEN, anchor=tk.W, bg="#BDC3C7")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # ======================= 主画面展示区 =======================
        self.canvas_width, self.canvas_height = 960, 600
        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg="black")
        self.canvas.pack(pady=10)
    
    def on_conf_change(self, val):
        """当用户拖动置信度滑动条时触发。如果是处理静态图片，则立刻重新推理刷新画面"""
        if not self.is_playing and self.current_frame is not None:
            # 停止流说明是静态图，立刻重绘
            res_frame = self.inference_frame(self.current_frame)
            self.show_frame(res_frame)

    def load_image(self):
        """导入一张静态图片"""
        self.stop_stream()
        file_path = filedialog.askopenfilename(title="选择农田图片", filetypes=[("Image Files", "*.jpg *.jpeg *.png"), ("All Files", "*.*")])
        if file_path:
            img = cv2.imread(file_path)
            if img is not None:
                self.current_frame = img
                # 推理并显示
                res_frame = self.inference_frame(self.current_frame)
                self.show_frame(res_frame)
                self.status_bar.config(text=f"当前展示静态图片: {os.path.basename(file_path)}")

    def load_video(self):
        """导入一个测试视频"""
        self.stop_stream()
        file_path = filedialog.askopenfilename(title="选择果园巡检视频", filetypes=[("Video Files", "*.mp4 *.avi *.mkv"), ("All Files", "*.*")])
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            self.is_playing = True
            self.status_bar.config(text=f"正在播放视频流: {os.path.basename(file_path)}")
            self.update_video_frame()

    def open_camera(self):
        """开启物理机器人的本地 USB/CSI 相机"""
        self.stop_stream()
        # 0 通常为默认的本地摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_bar.config(text="⚠️ 错误: 无法找到或打开本地摄像头 (ID: 0)")
            return
        self.is_playing = True
        self.status_bar.config(text="正在传输实时本地相机流...")
        self.update_video_frame()

    def stop_stream(self):
        """释放所有资源，清空画布"""
        self.is_playing = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None
        self.canvas.delete("all")
        self.current_frame = None
        self.status_bar.config(text="流已停止。")

    def update_video_frame(self):
        """Tkinter 事件循环：读取视频流中的下一帧并推理显示"""
        if self.is_playing and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                res_frame = self.inference_frame(frame)
                self.show_frame(res_frame)
                # 每隔约 30ms 拉取下一帧 (~33 FPS)
                self.after_id = self.root.after(30, self.update_video_frame)
            else:
                self.stop_stream()
                self.status_bar.config(text="视频播放结束。")

    def inference_frame(self, frame):
        """执行 YOLOv8-seg 核心推理"""
        current_conf = float(self.conf_slider.get())
        
        # 调用大模型，动态传入滑块指定的 conf 阈值。关闭 verbose 防止终端刷屏卡顿
        results = self.model(frame, conf=current_conf, retina_masks=True, verbose=False)[0]
        
        # 利用 internal plotter 画出包含 Mask 和 Box 的合成图
        annotated_frame = results.plot()
        return annotated_frame

    def show_frame(self, frame):
        """将 OpenCV 的 BGR 图像转换并自适应缩放渲染到 Tkinter 画布上"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        
        # 保持缩放比例适配 Canvas
        scale = min(self.canvas_width / w, self.canvas_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
        
        # 转换为 Tkinter 可识别的 Image 对象
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))
        
        self.canvas.delete("all")
        # 将图片放在画布正中间
        self.canvas.create_image(self.canvas_width // 2, self.canvas_height // 2, image=img_tk, anchor=tk.CENTER)
        
        # 避免被垃圾回收机制(GC)销毁
        self.canvas.image = img_tk

if __name__ == "__main__":
    # 以防环境丢失环境变量抛出 X11 错误
    root = tk.Tk()
    app = AgriRobotGUI(root, "🍎 智慧农业机器人视觉流控制台 v1.0")
    root.mainloop()