"""
OCR文本检测节点 - 用于视频文字后处理
"""
import torch
import numpy as np
import cv2
from PIL import Image
import time

from .PaddleOCR2Pytorch.pytorchocr.base_ocr_v20 import BaseOCRV20
from .PaddleOCR2Pytorch.tools.infer import pytorchocr_utility as utility
from .PaddleOCR2Pytorch.pytorchocr.data import create_operators, transform
from .PaddleOCR2Pytorch.pytorchocr.postprocess import build_post_process


class SimpleTextDetector(BaseOCRV20):
    """
    简化版文字检测器，专门用于DB算法的PP-OCRv5模型
    """
    def __init__(self, model_path, yaml_path, use_gpu=True, 
                 det_limit_side_len=960, det_limit_type='max',
                 det_db_thresh=0.3, det_db_box_thresh=0.6, 
                 det_db_unclip_ratio=1.5, use_dilation=False, 
                 det_db_score_mode='fast'):
        
        self.det_algorithm = 'DB'
        
        # 预处理配置
        pre_process_list = [{
            'DetResizeForTest': {
                'limit_side_len': det_limit_side_len,
                'limit_type': det_limit_type,
            }
        }, {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image', 'shape']
            }
        }]
        
        # 后处理配置
        postprocess_params = {
            'name': 'DBPostProcess',
            'thresh': det_db_thresh,
            'box_thresh': det_db_box_thresh,
            'max_candidates': 1000,
            'unclip_ratio': det_db_unclip_ratio,
            'use_dilation': use_dilation,
            'score_mode': det_db_score_mode
        }
        
        self.preprocess_op = create_operators(pre_process_list)
        self.postprocess_op = build_post_process(postprocess_params)
        
        # GPU配置
        self.use_gpu = torch.cuda.is_available() and use_gpu
        
        # 加载模型
        self.weights_path = model_path
        self.yaml_path = yaml_path
        network_config = utility.AnalysisConfig(self.weights_path, self.yaml_path)
        super(SimpleTextDetector, self).__init__(network_config)
        self.load_pytorch_weights(self.weights_path)
        self.net.eval()
        if self.use_gpu:
            self.net.cuda()
    
    def order_points_clockwise(self, pts):
        """按顺时针方向排序点"""
        xSorted = pts[np.argsort(pts[:, 0]), :]
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost
        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost
        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect
    
    def clip_det_res(self, points, img_height, img_width):
        """裁剪检测结果到图像范围内"""
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points
    
    def filter_tag_det_res(self, dt_boxes, image_shape):
        """过滤检测结果，去除过小的框"""
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes
    
    def predict(self, img):
        """对单张图像进行文字检测"""
        ori_im = img.copy()
        data = {'image': img}
        
        # 预处理
        data = transform(data, self.preprocess_op)
        img, shape_list = data
        if img is None:
            return np.array([]), 0
        
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        img = img.copy()
        
        st = time.time()
        
        # 推理
        with torch.no_grad():
            inp = torch.from_numpy(img)
            if self.use_gpu:
                inp = inp.cuda()
            outputs = self.net(inp)
        
        # 后处理
        preds = {'maps': outputs['maps'].cpu().numpy()}
        post_result = self.postprocess_op(preds, shape_list)
        dt_boxes = post_result[0]['points']
        
        # 过滤结果
        dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)
        
        et = time.time()
        return dt_boxes, et - st


def calculate_iou(box1, box2, frame_height, frame_width):
    """计算两个不规则四边形的 IoU"""
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
        
    iou = intersection_area / union_area
    return iou


def filter_product_text_boxes(boxes_v1, boxes_v2, depth_frame, frame_height, frame_width, 
                              iou_threshold=0.1, depth_threshold=50):
    """
    根据规则过滤掉 video1 中的商品文字框，返回非商品文字框。
    
    规则:
    1. v1中的框，若在v2中有高IoU匹配，则一定不是商品框。
    2. v1中的框，若在v2中没有高IoU匹配，并且在v1中的深度图区域的均值小于阈值（背景区域）则一定是商品框。
    """
    non_product_boxes = []   
    for box1 in boxes_v1:
        # 规则 1: 优先判断
        has_match_in_v2 = False
        for box2 in boxes_v2:
            iou = calculate_iou(box1, box2, frame_height, frame_width)
            if iou > iou_threshold:
                has_match_in_v2 = True
                break
        
        if has_match_in_v2:
            non_product_boxes.append(box1)
            continue

        # 规则 2: 深度图判断 (仅当无v2匹配时)
        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        cv2.fillPoly(mask, [box1.astype(np.int32)], 255)
        
        depth_roi = cv2.mean(depth_frame, mask=mask)[0]
        
        # 深度大于等于threshold视为非商品框（保留）
        if depth_roi >= depth_threshold:
            non_product_boxes.append(box1)
            
    return non_product_boxes


def tensor_to_numpy(tensor_image):
    """将ComfyUI的tensor图像转换为numpy数组 (BGR格式用于OpenCV)"""
    # tensor_image shape: [H, W, C], range [0, 1], RGB
    image_np = (tensor_image.cpu().numpy() * 255).astype(np.uint8)
    # 转换为BGR供OpenCV使用
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    return image_bgr


def convert_depth_to_gray(depth_bgr):
    """将深度图转换为灰度图"""
    if len(depth_bgr.shape) == 3:
        # 如果是彩色图，转换为灰度图
        return cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2GRAY)
    return depth_bgr


def numpy_to_tensor(image_bgr):
    """将numpy数组(BGR)转换回ComfyUI的tensor格式"""
    # 转换回RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # 归一化到 [0, 1]
    tensor = torch.from_numpy(image_rgb.astype(np.float32) / 255.0)
    return tensor


# ============= ComfyUI 节点定义 =============

class OCRModelLoader:
    """
    OCR模型加载节点 - 加载PP-OCRv5文字检测模型
    """
    
    # 类变量用于缓存已加载的模型，确保只加载一次
    _model_cache = {}
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "", "multiline": False}),
                "yaml_path": ("STRING", {"default": "", "multiline": False}),
                "use_gpu": ("BOOLEAN", {"default": True}),
                "det_limit_side_len": ("INT", {"default": 960, "min": 320, "max": 2048, "step": 32}),
                "det_db_thresh": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "det_db_box_thresh": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "det_db_unclip_ratio": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("OCR_MODEL",)
    RETURN_NAMES = ("ocr_model",)
    FUNCTION = "load_model"
    CATEGORY = "CRC/OCR"
    
    def load_model(self, model_path, yaml_path, use_gpu, det_limit_side_len, 
                   det_db_thresh, det_db_box_thresh, det_db_unclip_ratio):
        """加载OCR模型，使用缓存避免重复加载"""
        
        cache_key = f"{model_path}_{yaml_path}_{use_gpu}_{det_limit_side_len}_{det_db_thresh}_{det_db_box_thresh}_{det_db_unclip_ratio}"
        
        if cache_key in self._model_cache:
            print("使用已缓存的OCR模型")
            return (self._model_cache[cache_key],)
        
        print(f"正在加载 PP-OCRv5 模型: {model_path}")
        
        model = SimpleTextDetector(
            model_path=model_path,
            yaml_path=yaml_path,
            use_gpu=use_gpu,
            det_limit_side_len=det_limit_side_len,
            det_db_thresh=det_db_thresh,
            det_db_box_thresh=det_db_box_thresh,
            det_db_unclip_ratio=det_db_unclip_ratio
        )
        
        self._model_cache[cache_key] = model
        print("OCR模型加载完成")
        
        return (model,)


class TextRegionTransfer:
    """
    文字区域转移节点 - 将原视频的文字区域转移到修复后的视频
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ocr_model": ("OCR_MODEL",),
                "source_images": ("IMAGE",),  # 原视频帧序列
                "target_images": ("IMAGE",),  # 修复后视频帧序列
                "depth_images": ("IMAGE",),   # 深度图序列
                "iou_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "depth_threshold": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 255.0, "step": 1.0}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output_images",)
    FUNCTION = "process_frames"
    CATEGORY = "CRC/OCR"
    
    def process_frames(self, ocr_model, source_images, target_images, depth_images, 
                      iou_threshold, depth_threshold):
        """
        逐帧处理视频序列，将文字区域从源视频转移到目标视频
        """
        
        # 获取帧数
        num_frames = source_images.shape[0]
        
        if target_images.shape[0] != num_frames or depth_images.shape[0] != num_frames:
            raise ValueError(f"三个视频序列的帧数必须一致: source={num_frames}, target={target_images.shape[0]}, depth={depth_images.shape[0]}")
        
        print(f"开始处理 {num_frames} 帧")
        
        output_frames = []
        
        for frame_idx in range(num_frames):
            # 转换为numpy格式 (BGR for OpenCV)
            frame1 = tensor_to_numpy(source_images[frame_idx])
            frame2 = tensor_to_numpy(target_images[frame_idx])
            depth_frame = tensor_to_numpy(depth_images[frame_idx])
            
            # 统一尺寸到 frame2（目标视频）的尺寸
            height, width = frame2.shape[:2]
            
            # 如果 frame1 尺寸不匹配，resize
            if frame1.shape[:2] != (height, width):
                frame1 = cv2.resize(frame1, (width, height), interpolation=cv2.INTER_LINEAR)
            
            # 如果 depth_frame 尺寸不匹配，resize
            if depth_frame.shape[:2] != (height, width):
                depth_frame = cv2.resize(depth_frame, (width, height), interpolation=cv2.INTER_LINEAR)
            
            # 将深度图转换为灰度图（用于计算深度均值）
            depth_gray = convert_depth_to_gray(depth_frame)
            
            final_frame = frame2.copy()
            
            # 文字检测
            all_boxes1, _ = ocr_model.predict(frame1)
            all_boxes2, _ = ocr_model.predict(frame2)
            
            # 过滤商品文字框（使用灰度深度图）
            non_product_boxes1 = filter_product_text_boxes(
                all_boxes1, 
                all_boxes2, 
                depth_gray,
                height, 
                width, 
                iou_threshold,
                depth_threshold
            )
            
            # 处理非商品文字框
            if len(non_product_boxes1) > 0:
                for box1 in non_product_boxes1:
                    box1_int = box1.astype(np.int32)
                    
                    # 寻找最佳匹配
                    best_match_box = None
                    max_iou = 0.0
                    for box2 in all_boxes2:
                        iou = calculate_iou(box1, box2, height, width)
                        if iou > max_iou:
                            max_iou = iou
                            best_match_box = box2
                    
                    # 创建掩码
                    paste_mask = None
                    if max_iou > iou_threshold:
                        # 存在匹配框，掩码为并集
                        box2_int = best_match_box.astype(np.int32)
                        mask1_temp = np.zeros((height, width), dtype=np.uint8)
                        mask2_temp = np.zeros((height, width), dtype=np.uint8)
                        cv2.fillPoly(mask1_temp, [box1_int], 255)
                        cv2.fillPoly(mask2_temp, [box2_int], 255)
                        paste_mask = cv2.bitwise_or(mask1_temp, mask2_temp)
                    else:
                        # 不存在匹配框，掩码为box1本身
                        paste_mask = np.zeros((height, width), dtype=np.uint8)
                        cv2.fillPoly(paste_mask, [box1_int], 255)
                    
                    # 执行粘贴操作
                    if paste_mask is not None:
                        mask_inv = cv2.bitwise_not(paste_mask)
                        bg_cleared = cv2.bitwise_and(final_frame, final_frame, mask=mask_inv)
                        fg_extracted = cv2.bitwise_and(frame1, frame1, mask=paste_mask)
                        final_frame = cv2.add(bg_cleared, fg_extracted)
            
            # 转换回tensor格式
            output_tensor = numpy_to_tensor(final_frame)
            output_frames.append(output_tensor)
            
            if (frame_idx + 1) % 10 == 0:
                print(f"已处理 {frame_idx + 1}/{num_frames} 帧")
        
        # 合并为batch
        output_batch = torch.stack(output_frames, dim=0)
        
        print("处理完成")
        return (output_batch,)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "OCRModelLoader": OCRModelLoader,
    "TextRegionTransfer": TextRegionTransfer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OCRModelLoader": "OCR Model Loader",
    "TextRegionTransfer": "Text Region Transfer",
}

