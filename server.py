from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import torch
import json
import datetime
import os
import time
from threading import Thread
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，支持无桌面环境保存图片
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import TextIteratorStreamer
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.visualize import visualize_depth
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results

# 导入重构后的ros_api（确保包含TrackingDataClient和TrackingResult等类型）
import ros_api
from ros_api import TrackingResult, CameraIntrinsics, CameraPose  # 导入强类型数据结构

# ===================== 配置常量 =====================
# 模型配置
MODEL_QWEN_8B = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_QWEN_4B = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_QWEN_2B = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_SMOLVLM = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
MODEL_DA3_LARGE = "depth-anything/DA3-LARGE-1.1"
MODEL_DA3_NESTED = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
MODEL_VLM_DEFAULT = MODEL_QWEN_4B
MODEL_DA3_DEFAULT = MODEL_DA3_LARGE
MODEL_SAM3_PATH = "/home/seqn/sam3/sam3.pt"  # 根据自己的模型权重地址修改

# 网络配置
ROS_SERVER_IP = "127.0.0.1"
ROS_SERVER_PORT = 51121
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 51122

# 生成配置
MAX_NEW_TOKENS = 256
IMAGE_SAVE_PREFIX = "images/image_"
DEPTH_PLOT_SAVE_PREFIX = "images/depth_comparison_"
DA3_DEPTH_SAVE_PREFIX = "images/da3_depth_"
DA3_DEPTH_WITH_KEYPOINTS_SAVE_PREFIX = "images/da3_depth_with_keypoints_"

# 确保图像保存目录存在
os.makedirs("images", exist_ok=True)

# ===================== 全局对象 =====================
app = None
processor = None
model_vlm = None
model_da3 = None
processor_sam3 = None

def init_tracking_client(enable_log: bool = False) -> ros_api.TrackingDataClient:
    """
    新建并返回ROS跟踪数据客户端实例（每次调用新建连接，适配重构版客户端）
    :param enable_log: 是否启用客户端日志（默认关闭，避免与Flask日志冲突）
    :return: TrackingDataClient实例（无需提前连接，闭环方法内部处理连接）
    """
    try:
        # 实例化重构版ros_api客户端（仅初始化，不提前连接）
        client = ros_api.TrackingDataClient(
            server_ip=ROS_SERVER_IP,
            port=ROS_SERVER_PORT,
            enable_log=enable_log  # 关闭客户端日志，由Flask统一输出
        )
        return client
    except Exception as e:
        error_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        print(f"[{error_time}] ❌ 跟踪客户端创建失败: {str(e)}")
        raise Exception(f"跟踪客户端创建失败: {str(e)}")

def get_image_from_ros(client: ros_api.TrackingDataClient, timestamp: str) -> tuple[Image.Image, str, np.ndarray, np.ndarray, np.ndarray] | tuple[None, str, None, None, None]:
    """
    从传入的ROS客户端获取图像、点云数据并转换为PIL Image（适配闭环方法+强类型返回值）
    :param client: 重构版TrackingDataClient实例
    :param timestamp: 时间戳（用于生成图像文件名）
    :return: (PIL图像对象, 图像保存路径, 相机坐标点云, 世界坐标点云, 原始OpenCV图像) 或错误元组
    """
    if not client:
        return None, "ROS客户端实例无效", None, None, None
    
    try:
        # ============== 核心重构：调用闭环方法一键完成全流程 ==============
        print(f"[{timestamp}] 🔍 开始执行ROS数据闭环获取流程...")
        tracking_result: TrackingResult | None = client.complete_tracking_pipeline()
        
        # 校验闭环方法返回结果（强类型对象）
        if not tracking_result:
            return None, "ROS数据闭环获取失败（连接/解析/请求任一环节出错）", None, None, None
        
        # ============== 从强类型TrackingResult中提取数据（替换原字典取值） ==============
        # 1. 提取OpenCV图像（保留原始图像，用于后续匹配深度图尺寸）
        cv_image = tracking_result.current_image
        if cv_image is None or not isinstance(cv_image, np.ndarray):
            return None, "从ROS闭环结果中提取图像失败", None, None, None
        
        # 2. 提取ORB-SLAM3点云数据（相机坐标/世界坐标）
        camera_point_cloud = tracking_result.tracked_points_camera
        world_point_cloud = tracking_result.tracked_points_world
        
        # 3. OpenCV图像转换为PIL Image（CV2: BGR → PIL: RGB）
        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv_image_rgb)
        
        # 4. 保存图像到本地
        image_save_path = f"{IMAGE_SAVE_PREFIX}{timestamp}.jpg"
        pil_image.save(image_save_path)
        print(f"[{timestamp}] 💾 保存处理后图像到: {image_save_path}（尺寸：{pil_image.size}，模式：{pil_image.mode}）")
        print(f"[{timestamp}] 📊 ROS数据闭环获取统计：总接收{tracking_result.total_recv_size}字节，解析耗时{tracking_result.parse_cost_ms:.2f}ms")
        
        return pil_image, image_save_path, camera_point_cloud, world_point_cloud, cv_image
    
    except Exception as e:
        error_msg = f"从ROS闭环结果处理数据失败: {str(e)}"
        print(f"[{timestamp}] ❌ {error_msg}")
        return None, error_msg, None, None, None

# ===================== 模型加载模块（无修改，保持原有逻辑） =====================
def load_model_vlm(model_path: str = MODEL_VLM_DEFAULT):
    """
    加载并编译AI模型
    :param model_path: 模型路径/名称
    """
    global processor, model_vlm
    try:
        # 1. 加载处理器
        print(f"{time.time()} > 加载模型处理器: {model_path}...", flush=True)
        processor = AutoProcessor.from_pretrained(model_path)
        
        # 2. 加载模型
        print(f"{time.time()} > 加载模型权重...", flush=True)
        model_vlm = AutoModelForImageTextToText.from_pretrained(
            model_path,
            load_in_8bit=True
        )
        
        # 3. 编译模型（优化推理速度）
        print(f"{time.time()} > 编译模型...", flush=True)
        model_vlm = torch.compile(model_vlm)
        
        print(f"{time.time()} > ✅ VLM 模型加载并编译完成！", flush=True)
    except Exception as e:
        raise Exception(f"模型加载失败: {str(e)}")

def load_model_da3(model_path=MODEL_DA3_DEFAULT):
    """加载DA3深度估计模型并初始化全局对象"""
    global model_da3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_da3 = DepthAnything3.from_pretrained(model_path)
    model_da3 = model_da3.to(device)
    model_da3.eval()
    print(f"{time.time()} > ✅ DA3 模型加载并编译完成！", flush=True)

def load_model_sam3(model_path=MODEL_SAM3_PATH):
    """加载SAM3模型并初始化全局对象"""
    global processor_sam3
    model = build_sam3_image_model(
        load_from_HF=False,
        checkpoint_path=model_path
    )
    processor_sam3 = Sam3Processor(model)
    print(f"{time.time()} > ✅ SAM3 模型加载并编译完成！", flush=True)

# ===================== 深度生成模块（无修改，保持原有逻辑） =====================
def generate_depth_map(image_path: str, target_shape: tuple[int, int]):
    """
    生成指定尺寸的深度图，使用INTER_CUBIC插值进行缩放，匹配原始图像尺寸
    :param image_path: 输入图像路径
    :param target_shape: 目标深度图尺寸 (h, w)（与原始图像一致）
    :return: 与目标尺寸匹配的深度图数组
    """
    global model_da3
    if model_da3 is None:
        raise Exception("DA3模型未初始化，请先调用load_model_da3()")
    
    prediction = model_da3.inference(
        image=[image_path],
        process_res=504,
        process_res_method="upper_bound_resize",
        export_dir=None,
        export_format="glb"
    )
    
    # 提取预测深度图并调整尺寸至目标形状，使用INTER_CUBIC插值保证精度
    depth_map = prediction.depth[0]
    depth_map_resized = cv2.resize(
        depth_map,
        (target_shape[1], target_shape[0]),  # cv2.resize参数为(w, h)，与target_shape (h, w)对应
        interpolation=cv2.INTER_CUBIC
    )
    
    return depth_map_resized

# ===================== 深度对比绘图模块（无修改，保持原有逻辑） =====================
def plot_depth_comparison(camera_point_cloud: np.ndarray, da3_depth_map: np.ndarray, timestamp: str, image_shape: tuple[int, int]):
    """
    严格根据camera_point_cloud的x/y（像素坐标）提取对应DA3深度，z轴为实际深度，绘制关系图并保存
    :param camera_point_cloud: 相机坐标点云 (N, 3)，x=像素w, y=像素h, z=ORB-SLAM3实际深度
    :param da3_depth_map: DA3生成的深度图 (h, w)
    :param timestamp: 时间戳，用于生成保存文件名
    :param image_shape: 原始图像尺寸 (h, w)，用于校验像素坐标有效性
    """
    # 1. 前置校验
    if camera_point_cloud is None or len(camera_point_cloud) == 0:
        print(f"[{timestamp}] ⚠️  无有效ORB-SLAM3点云数据，跳过绘图")
        return
    if da3_depth_map.shape != image_shape:
        print(f"[{timestamp}] ⚠️  深度图尺寸与图像尺寸不匹配，跳过绘图")
        return
    
    # 2. 提取点云数据，分离x（像素w/列）、y（像素h/行）、z（实际深度）
    pixel_w = camera_point_cloud[:, 0].astype(np.int32)  # x对应图像列（w）
    pixel_h = camera_point_cloud[:, 1].astype(np.int32)  # y对应图像行（h）
    orb_slam_depth = camera_point_cloud[:, 2]  # z轴为ORB-SLAM3实际深度（真实值）
    
    # 3. 过滤无效数据
    valid_mask = np.logical_and.reduce([
        orb_slam_depth > 0,  # 过滤无效深度（<=0）
        pixel_w >= 0,
        pixel_w < image_shape[1],  # 过滤超出图像宽度的像素坐标
        pixel_h >= 0,
        pixel_h < image_shape[0]   # 过滤超出图像高度的像素坐标
    ])
    
    # 4. 提取有效数据
    valid_pixel_w = pixel_w[valid_mask]
    valid_pixel_h = pixel_h[valid_mask]
    valid_orb_slam_depth = orb_slam_depth[valid_mask]
    
    if len(valid_orb_slam_depth) == 0:
        print(f"[{timestamp}] ⚠️  无有效像素坐标或深度数据，跳过绘图")
        return
    
    # 5. 严格根据有效像素坐标（h, w）提取DA3对应位置的深度值
    # da3_depth_map[pixel_h, pixel_w] 对应：行=pixel_h，列=pixel_w，与图像坐标一一对应
    valid_da3_depth = da3_depth_map[valid_pixel_h, valid_pixel_w]
    
    # 6. 再次过滤DA3无效深度（<=0）
    final_valid_mask = valid_da3_depth > 0
    final_orb_slam_depth = valid_orb_slam_depth[final_valid_mask]
    final_da3_depth = valid_da3_depth[final_valid_mask]
    
    if len(final_orb_slam_depth) == 0:
        print(f"[{timestamp}] ⚠️  无有效DA3深度数据，跳过绘图")
        return
    
    # 7. 绘制关系图（保留一一对应关系，不丢失点云信息）
    plt.figure(figsize=(12, 6))
    
    # 子图1：散点图（展示两者一一对应的深度关系，核心对比）
    plt.subplot(1, 2, 1)
    plt.scatter(final_orb_slam_depth, final_da3_depth, alpha=0.7, s=8, c='royalblue')
    # 添加对角线（理想情况下，两者深度应落在对角线上）
    max_depth = np.max([np.max(final_orb_slam_depth), np.max(final_da3_depth)])
    plt.plot([0, max_depth], [0, max_depth], 'r--', alpha=0.8, label="Ideal Match")
    plt.xlabel("ORB-SLAM3 True Depth (m)")
    plt.ylabel("DA3 Predicted Depth (m)")
    plt.title("ORB-SLAM3 vs DA3 Depth (Pixel-wise Match)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2：差值直方图（展示两者深度误差分布）
    plt.subplot(1, 2, 2)
    depth_diff = final_orb_slam_depth - final_da3_depth
    plt.hist(depth_diff, bins=50, alpha=0.7, color='purple', edgecolor='black', linewidth=0.5)
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.8, label="Zero Error")
    plt.xlabel("Depth Error (ORB-SLAM3 - DA3) (m)")
    plt.ylabel("Point Count")
    plt.title("Depth Error Distribution Histogram")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. 调整布局并保存图片
    plt.tight_layout()
    plot_save_path = f"{DEPTH_PLOT_SAVE_PREFIX}{timestamp}.png"
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 9. 打印统计信息
    print(f"[{timestamp}] 💾 深度对比图保存到: {plot_save_path}")
    print(f"[{timestamp}] 📊 有效对比点数量: {len(final_orb_slam_depth)}")
    print(f"[{timestamp}] 📊 平均深度误差: {np.mean(np.abs(depth_diff)):.6f} m")
    print(f"[{timestamp}] 📊 均方根误差: {np.sqrt(np.mean(depth_diff ** 2)):.6f} m")

def save_da3_depth_with_ros_keypoints(da3_depth_map: np.ndarray, camera_point_cloud: np.ndarray, timestamp: str, image_shape: tuple[int, int]):
    """
    1. 保存原始DA3深度图（着色可视化）
    2. 在深度图上叠加ROS关键点：白色边框 + ORB-SLAM3真实深度色填充，便于直观判断深度误差
    :param da3_depth_map: DA3生成的深度图 (h, w)
    :param camera_point_cloud: 相机坐标点云 (N, 3)，x=像素w, y=像素h, z=ORB-SLAM3实际深度
    :param timestamp: 时间戳，用于生成保存文件名
    :param image_shape: 原始图像尺寸 (h, w)，用于校验像素坐标有效性
    """
    # 前置校验
    if da3_depth_map is None or da3_depth_map.shape != image_shape:
        print(f"[{timestamp}] ⚠️  DA3深度图无效或尺寸不匹配，跳过深度图保存和关键点叠加")
        return
    
    # 步骤1：可视化DA3深度图（着色，与depth_anything_3风格一致）
    da3_depth_viz = visualize_depth(da3_depth_map, cmap='plasma')  # plasma着色方案
    da3_depth_viz = (da3_depth_viz * 255).astype(np.uint8)  # 归一化值转255级RGB
    if len(da3_depth_viz.shape) == 2:  # 灰度图转RGB
        da3_depth_viz = cv2.cvtColor(da3_depth_viz, cv2.COLOR_GRAY2RGB)
    
    # 步骤2：保存原始着色DA3深度图
    da3_depth_save_path = f"{DA3_DEPTH_SAVE_PREFIX}{timestamp}.png"
    cv2.imwrite(da3_depth_save_path, da3_depth_viz)
    print(f"[{timestamp}] 💾 原始DA3着色深度图保存到: {da3_depth_save_path}")
    
    # 步骤3：叠加ROS关键点（白色边框 + ORB-SLAM3真实深度色填充）
    da3_depth_with_keypoints = da3_depth_viz.copy()
    if camera_point_cloud is not None and len(camera_point_cloud) > 0:
        # 提取点云像素坐标和ORB-SLAM3真实深度值
        pixel_w = camera_point_cloud[:, 0].astype(np.int32)
        pixel_h = camera_point_cloud[:, 1].astype(np.int32)
        orb_slam_depth = camera_point_cloud[:, 2]  # 提取ORB-SLAM3真实深度（核心：从这里获取颜色映射依据）
        
        # 过滤有效像素坐标和有效深度值
        valid_mask = np.logical_and.reduce([
            orb_slam_depth > 0,
            pixel_w >= 0,
            pixel_w < image_shape[1],
            pixel_h >= 0,
            pixel_h < image_shape[0]
        ])
        
        valid_pixel_w = pixel_w[valid_mask]
        valid_pixel_h = pixel_h[valid_mask]
        valid_orb_slam_depth = orb_slam_depth[valid_mask]  # 有效ORB-SLAM3真实深度值
        
        if len(valid_pixel_w) > 0:
            # 关键点参数：外圆（白色边框）半径3，内圆（ORB真实深度色）半径2
            outer_radius = 3
            inner_radius = 2
            white_color = (255, 255, 255)  # 白色边框（BGR格式）
            percentile = 2  # 与visualize_depth函数默认百分位保持一致
            
            # ============== 核心修复：完全对齐visualize_depth的归一化逻辑 ==============
            # 步骤1：复制数据避免修改原始数组（与visualize_depth保持一致）
            orb_depth_processed = valid_orb_slam_depth.copy()
            
            # 步骤2：有效深度取倒数（visualize_depth核心逻辑：depth[valid_mask] = 1 / depth[valid_mask]）
            orb_valid_mask = orb_depth_processed > 0
            orb_depth_processed[orb_valid_mask] = 1 / orb_depth_processed[orb_valid_mask]
            
            # 步骤3：计算百分位对应的min/max（与visualize_depth逻辑一致）
            if orb_valid_mask.sum() <= 10:
                orb_depth_min = 0
                orb_depth_max = 0
            else:
                orb_depth_min = np.percentile(orb_depth_processed[orb_valid_mask], percentile)
                orb_depth_max = np.percentile(orb_depth_processed[orb_valid_mask], 100 - percentile)
            
            # 步骤4：避免min/max相等（防止除零错误，与visualize_depth逻辑一致）
            if orb_depth_min == orb_depth_max:
                orb_depth_min = orb_depth_min - 1e-6
                orb_depth_max = orb_depth_max + 1e-6
            
            # 步骤5：归一化到[0,1]范围（与visualize_depth逻辑一致）
            normalized_orb_depth = ((orb_depth_processed - orb_depth_min) / (orb_depth_max - orb_depth_min)).clip(0, 1)
            
            # 步骤6：数值翻转（visualize_depth核心逻辑：depth = 1 - depth）
            normalized_orb_depth = 1 - normalized_orb_depth
            # ============== 归一化逻辑修复完成 ==============
            
            # 兼容高版本Matplotlib，获取plasma色卡
            plasma_cmap = matplotlib.colormaps['plasma']  # 与DA3深度图着色色卡保持一致
            
            for idx, (w, h) in enumerate(zip(valid_pixel_w, valid_pixel_h)):
                # 1. 先画白色实心外圆（作为边框，醒目易识别）
                cv2.circle(
                    img=da3_depth_with_keypoints,
                    center=(w, h),
                    radius=outer_radius,
                    color=white_color,
                    thickness=-1  # 实心填充
                )
                
                # 2. 提取当前ORB-SLAM3真实深度对应的RGB颜色（核心修改：从ORB数据提取）
                norm_depth = normalized_orb_depth[idx]
                orb_rgb = plasma_cmap(norm_depth)[:3]  # 获取plasma色卡对应的RGB值（0~1范围）
                orb_rgb_255 = (np.array(orb_rgb) * 255).astype(np.uint8)  # 转换为0~255范围
                
                # 修复color非数值错误：转换为Python原生整数元组
                orb_bgr = (
                    int(orb_rgb_255[2]),
                    int(orb_rgb_255[1]),
                    int(orb_rgb_255[0])
                )
                
                # 4. 再画ORB真实深度色实心内圆（与DA3深度图色卡一致，便于对比误差）
                cv2.circle(
                    img=da3_depth_with_keypoints,
                    center=(w, h),
                    radius=inner_radius,
                    color=orb_bgr,
                    thickness=-1
                )
            
            # 添加关键点数量标注
            cv2.putText(
                img=da3_depth_with_keypoints,
                text=f"Valid ROS Keypoints: {len(valid_pixel_w)}",
                org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=white_color,
                thickness=2
            )
    
    # 步骤4：保存叠加关键点后的DA3深度图
    da3_depth_keypoints_save_path = f"{DA3_DEPTH_WITH_KEYPOINTS_SAVE_PREFIX}{timestamp}.png"
    cv2.imwrite(da3_depth_keypoints_save_path, da3_depth_with_keypoints)
    print(f"[{timestamp}] 💾 叠加ROS关键点的DA3深度图保存到: {da3_depth_keypoints_save_path}")

# ===================== 文本生成模块（无修改，保持原有逻辑） =====================
def generate_text_stream(text_query: str, image_path: str, timestamp: str):
    """
    流式生成文本响应
    :param text_query: 文本查询指令
    :param image_path: 图像保存路径
    :param timestamp: 时间戳
    """
    global processor, model_vlm
    if not processor or not model_vlm:
        yield f"data: {json.dumps({'text': '❌ 模型未加载完成'})}\n\n"
        return
    
    try:
        print(f"[{timestamp}] 🤖 开始生成文本，查询: '{text_query}'")
        
        # 1. 构建对话消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": image_path},
                    {"type": "text", "text": text_query}
                ]
            },
        ]
        
        # 2. 应用聊天模板并编码
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model_vlm.device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
        
        # 3. 初始化流式生成器
        streamer = TextIteratorStreamer(
            processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        # 4. 构建生成参数
        generation_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            num_beams=1
        )
        
        # 5. 启动独立线程执行生成
        thread = Thread(target=model_vlm.generate, kwargs=generation_kwargs)
        thread.start()
        
        # 6. 流式返回生成结果
        for new_text in streamer:
            if new_text:
                yield f"data: {json.dumps({'text': new_text})}\n\n"
        
        print(f"[{timestamp}] ✅ 文本生成完成")
    except Exception as e:
        error_msg = f"文本生成失败: {str(e)}"
        yield f"data: {json.dumps({'text': f'❌ {error_msg}'})}\n\n"

# ===================== Flask接口模块（仅适配客户端调用重构，其余不变） =====================
def init_flask_app():
    """
    初始化Flask应用
    """
    global app
    app = Flask(__name__)
    CORS(app)  # 允许跨域请求

    @app.route('/generate', methods=['POST'])
    def generate():
        # 1. 解析请求参数
        data = request.json or {}
        text_query = data.get('text', 'Describe this image')
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # 2. 每次请求新建ROS客户端（重构版，无需提前连接）
            print(f"[{timestamp}] 🔍 新建ROS客户端实例...")
            ros_client = init_tracking_client(enable_log=False)
            if not ros_client:
                raise Exception("新建ROS客户端实例失败")
            
            # 3. 传入新建的ros_client，从ROS获取图像和点云数据（适配闭环方法）
            print(f"[{timestamp}] 🔍 开始从ROS获取图像和点云数据...")
            pil_image, image_path, camera_point_cloud, world_point_cloud, cv_image = get_image_from_ros(ros_client, timestamp)
            if not pil_image:
                raise Exception(image_path)  # image_path此时为错误信息
            
            # 后续逻辑保持不变
            # 3. 获取原始图像尺寸（h, w），用于匹配深度图尺寸
            image_shape = cv_image.shape[:2]  # (h, w)
            print(f"[{timestamp}] 📏 原始图像尺寸: {image_shape}，准备生成对应深度图...")
            
            # 4. 生成DA3深度图（匹配原始图像尺寸）
            print(f"[{timestamp}] 📊 开始生成DA3深度图...")
            da3_depth_map = generate_depth_map(image_path, image_shape)
            
            # 5. 绘制并保存深度对比图（严格一一对应像素坐标）
            plot_depth_comparison(camera_point_cloud, da3_depth_map, timestamp, image_shape)
            save_da3_depth_with_ros_keypoints(da3_depth_map, camera_point_cloud, timestamp, image_shape)
            
            # 6. 流式返回生成结果
            return Response(generate_text_stream(text_query, image_path, timestamp), 
                            mimetype='text/event-stream')
        
        except Exception as e:
            error_msg = f"[{timestamp}] ❌ 错误: {str(e)}"
            print(error_msg)
            return jsonify({'error': error_msg}), 500

# ===================== 主程序入口（无修改，保持原有逻辑） =====================
def main():
    """主程序：协调各模块初始化，启动服务"""
    try:
        # 1. 初始化Flask应用
        init_flask_app()
        
        # 2. 加载各类AI模型
        load_model_vlm(MODEL_VLM_DEFAULT)
        load_model_da3(MODEL_DA3_DEFAULT)
        # load_model_sam3(MODEL_SAM3_PATH)
        
        # 3. 启动Flask服务
        print(f"\n[{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}] 🚀 Flask服务启动中，地址: http://{FLASK_HOST}:{FLASK_PORT}")
        app.run(
            host=FLASK_HOST,
            port=FLASK_PORT,
            threaded=True,
            debug=False  # 生产环境关闭debug
        )
    except Exception as e:
        print(f"[{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}] ❌ 程序启动失败: {str(e)}")
        os._exit(1)

if __name__ == '__main__':
    main()