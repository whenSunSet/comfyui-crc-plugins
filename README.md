# ComfyUI-CRC-Plugins

ComfyUI 自定义插件，用于视频文字对齐后处理。

## 功能

本插件提供两个节点：

### 1. PaddleOCR Model Loader
加载 PaddleOCR 模型用于文字检测和识别。

**输入参数:**
- `det_model_dir`: 检测模型路径
- `rec_model_dir`: 识别模型路径  
- `lang`: 语言选择（ch/en/japan/korean/german/french）

**输出:**
- `PADDLE_OCR_MODEL`: 模型实例

**注意:** 模型会被缓存，相同配置只加载一次。

### 2. Video Text Align Processor
对视频帧进行文字对齐后处理。

**输入参数:**
- `model`: PaddleOCR 模型实例
- `source_frames`: 原始视频帧序列（IMAGE类型）
- `target_frames`: 修复后视频帧序列（IMAGE类型）
- `depth_frames`: 深度图视频帧序列（IMAGE类型）
- `iou_threshold`: IoU 阈值（默认0.1）
- `depth_threshold`: 深度阈值（默认50.0）

**输出:**
- `IMAGE`: 处理后的视频帧序列

## 安装

1. 将本插件文件夹放入 ComfyUI 的 `custom_nodes` 目录
2. 安装依赖：
```bash
pip install -r requirements.txt
```
3. 重启 ComfyUI

## 使用流程

1. 使用 KJNodes 的 `Load Video` 节点加载三个视频（原始视频、修复后视频、深度视频）
2. 使用 `PaddleOCR Model Loader` 加载 OCR 模型
3. 将视频帧和模型连接到 `Video Text Align Processor` 节点
4. 使用 KJNodes 的 `Video Combine` 节点将输出帧合并为视频

## 原理

该插件实现视频文字区域的智能对齐：

1. 检测原始视频和修复后视频中的文字框
2. 根据 IoU 和深度信息过滤商品文字框
3. 将原始视频的非商品文字区域粘贴到修复后的视频上
4. 保持文字清晰度和一致性

