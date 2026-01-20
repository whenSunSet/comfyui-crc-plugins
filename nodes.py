import torch
import numpy as np
import cv2
from paddleocr import PaddleOCR
import os
from comfy.utils import ProgressBar

class PaddleOCRModelLoader:
    """加载 PaddleOCR 模型节点"""
    
    _model_cache = None  # 类变量缓存模型实例
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "det_model_dir": ("STRING", {
                    "default": "/path/to/PP-OCRv5_server_det_infer",
                    "multiline": False
                }),
                "rec_model_dir": ("STRING", {
                    "default": "/path/to/PP-OCRv5_server_rec_infer", 
                    "multiline": False
                }),
                "lang": (["ch", "en", "japan", "korean", "german", "french"],
                    {"default": "ch"}
                ),
            }
        }
    
    RETURN_TYPES = ("PADDLE_OCR_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "CRC-Plugins"
    
    def load_model(self, det_model_dir, rec_model_dir, lang):
        # 生成缓存key
        cache_key = f"{det_model_dir}_{rec_model_dir}_{lang}"
        
        # 如果模型已缓存且配置相同，直接返回
        if PaddleOCRModelLoader._model_cache is not None:
            cached_key = getattr(PaddleOCRModelLoader._model_cache, '_cache_key', None)
            if cached_key == cache_key:
                print("使用已缓存的 PaddleOCR 模型")
                return (PaddleOCRModelLoader._model_cache,)
        
        # 加载新模型
        print(f"正在加载 PaddleOCR 模型...")
        print(f"检测模型路径: {det_model_dir}")
        print(f"识别模型路径: {rec_model_dir}")
        
        model = PaddleOCR(
            det_model_dir=det_model_dir,
            rec_model_dir=rec_model_dir,
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )
        
        # 保存缓存key到模型对象
        model._cache_key = cache_key
        
        # 缓存模型
        PaddleOCRModelLoader._model_cache = model
        print("PaddleOCR 模型加载完成")
        
        return (model,)


class VideoTextAlignProcessor:
    """视频文字对齐后处理节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PADDLE_OCR_MODEL",),
                "source_frames": ("IMAGE",),  # video1 原始视频帧
                "target_frames": ("IMAGE",),  # video2 修复后视频帧
                "depth_frames": ("IMAGE",),   # depth 深度图视频帧
                "iou_threshold": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "depth_threshold": ("FLOAT", {
                    "default": 50.0,
                    "min": 0.0,
                    "max": 255.0,
                    "step": 1.0
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "CRC-Plugins"
    
    def process(self, model, source_frames, target_frames, depth_frames, 
                iou_threshold, depth_threshold):
        """
        处理视频帧
        Args:
            model: PaddleOCR 模型实例
            source_frames: 原始视频帧 [B, H, W, C] tensor
            target_frames: 目标视频帧 [B, H, W, C] tensor  
            depth_frames: 深度图帧 [B, H, W, C] tensor
            iou_threshold: IoU 阈值
            depth_threshold: 深度阈值
        Returns:
            处理后的视频帧 [B, H, W, C] tensor
        """
        # 转换为 numpy 数组
        source_np = (source_frames.cpu().numpy() * 255).astype(np.uint8)
        target_np = (target_frames.cpu().numpy() * 255).astype(np.uint8)
        depth_np = (depth_frames.cpu().numpy() * 255).astype(np.uint8)
        
        # 检查所有输入的帧数，使用最小值
        batch_size = min(source_np.shape[0], target_np.shape[0], depth_np.shape[0])
        
        # 如果帧数不一致，打印警告
        if not (source_np.shape[0] == target_np.shape[0] == depth_np.shape[0]):
            print(f"警告: 输入视频帧数不一致!")
            print(f"  source_frames: {source_np.shape[0]} 帧")
            print(f"  target_frames: {target_np.shape[0]} 帧")
            print(f"  depth_frames: {depth_np.shape[0]} 帧")
            print(f"  将处理前 {batch_size} 帧")
        
        output_frames = []
        pbar = ProgressBar(batch_size)
        
        for i in range(batch_size):
            print("开始处理第", i, "帧")
            frame1 = source_np[i]  # [H, W, C]
            frame2 = target_np[i]
            depth_frame = depth_np[i]
            
            # 获取当前帧的实际尺寸
            height, width = frame2.shape[0], frame2.shape[1]
            
            # 确保所有帧尺寸一致
            if frame1.shape[:2] != (height, width):
                frame1 = cv2.resize(frame1, (width, height))
            
            # 确保深度图尺寸匹配并转换为单通道
            depth_frame = cv2.resize(depth_frame, (width, height))
            if len(depth_frame.shape) == 3:
                depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_RGB2GRAY)
            
            # OCR 检测
            result1 = model.predict(input=frame1)[0]
            result2 = model.predict(input=frame2)[0]
            
            all_boxes1 = result1['dt_polys'] if result1 else []
            all_boxes2 = result2['dt_polys'] if result2 else []
            
            # 过滤商品文字框
            non_product_boxes1 = self.filter_product_text_boxes(
                all_boxes1, all_boxes2, depth_frame, 
                height, width, iou_threshold, depth_threshold
            )
            
            # 复制目标帧作为输出
            final_frame = frame2.copy()
            
            # 处理非商品框
            for box1 in non_product_boxes1:
                box1_int = box1.astype(np.int32)
                
                # 查找最佳匹配
                best_match_box = None
                max_iou = 0.0
                for box2 in all_boxes2:
                    iou = self.calculate_iou(box1, box2, height, width)
                    if iou > max_iou:
                        max_iou = iou
                        best_match_box = box2
                
                # 创建粘贴掩码
                paste_mask = None
                if max_iou > iou_threshold and best_match_box is not None:
                    # 并集掩码
                    box2_int = best_match_box.astype(np.int32)
                    mask1_temp = np.zeros((height, width), dtype=np.uint8)
                    mask2_temp = np.zeros((height, width), dtype=np.uint8)
                    cv2.fillPoly(mask1_temp, [box1_int], 255)
                    cv2.fillPoly(mask2_temp, [box2_int], 255)
                    paste_mask = cv2.bitwise_or(mask1_temp, mask2_temp)
                else:
                    # box1 本身作为掩码
                    paste_mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.fillPoly(paste_mask, [box1_int], 255)
                
                # 粘贴操作
                if paste_mask is not None:
                    mask_inv = cv2.bitwise_not(paste_mask)
                    bg_cleared = cv2.bitwise_and(final_frame, final_frame, mask=mask_inv)
                    fg_extracted = cv2.bitwise_and(frame1, frame1, mask=paste_mask)
                    final_frame = cv2.add(bg_cleared, fg_extracted)
            
            output_frames.append(final_frame)
            pbar.update(1)
        
        # 转回 tensor [B, H, W, C]
        output_np = np.stack(output_frames, axis=0)
        output_tensor = torch.from_numpy(output_np.astype(np.float32) / 255.0)
        
        return (output_tensor,)
    
    def filter_product_text_boxes(self, boxes_v1, boxes_v2, depth_frame, 
                                   frame_height, frame_width, 
                                   iou_threshold, depth_threshold):
        """过滤商品文字框"""
        non_product_boxes = []
        
        for box1 in boxes_v1:
            # 规则1: 检查是否在 v2 中有高 IoU 匹配
            has_match_in_v2 = False
            for box2 in boxes_v2:
                iou = self.calculate_iou(box1, box2, frame_height, frame_width)
                if iou > iou_threshold:
                    has_match_in_v2 = True
                    break
            
            if has_match_in_v2:
                non_product_boxes.append(box1)
                continue
            
            # 规则2: 深度图判断
            mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
            cv2.fillPoly(mask, [box1.astype(np.int32)], 255)
            depth_roi = cv2.mean(depth_frame, mask=mask)[0]
            
            if depth_roi >= depth_threshold:
                non_product_boxes.append(box1)
        
        return non_product_boxes
    
    def calculate_iou(self, box1, box2, frame_height, frame_width):
        """计算两个多边形的 IoU"""
        poly1 = box1.astype(np.int32)
        poly2 = box2.astype(np.int32)
        
        mask1 = np.zeros((frame_height, frame_width), dtype=np.uint8)
        mask2 = np.zeros((frame_height, frame_width), dtype=np.uint8)
        
        cv2.fillPoly(mask1, [poly1], 255)
        cv2.fillPoly(mask2, [poly2], 255)
        
        intersection_mask = cv2.bitwise_and(mask1, mask2)
        intersection_area = cv2.countNonZero(intersection_mask)
        
        area1 = cv2.contourArea(poly1)
        area2 = cv2.contourArea(poly2)
        union_area = area1 + area2 - intersection_area
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area


# 节点映射
NODE_CLASS_MAPPINGS = {
    "PaddleOCRModelLoader": PaddleOCRModelLoader,
    "VideoTextAlignProcessor": VideoTextAlignProcessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PaddleOCRModelLoader": "PaddleOCR Model Loader",
    "VideoTextAlignProcessor": "Video Text Align Processor",
}

