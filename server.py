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
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import TextIteratorStreamer
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.visualize import visualize_depth
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results

# 导入ros_api（确保ros_api.py与当前文件在同一目录）
import ros_api

# ===================== 配置常量（模块化：统一管理配置） =====================
# 模型配置
MODEL_QWEN_8B = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_QWEN_4B = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_QWEN_2B = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_SMOLVLM = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
MODEL_DA3_LARGE = "depth-anything/DA3-LARGE-1.1"
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
IMAGE_SAVE_PREFIX = "image_"

# ===================== 全局对象（模块化：延迟初始化，避免提前加载） =====================
app = None
processor = None
model_vlm = None
model_da3 = None
processor_sam3 = None
tracking_client = None

# ===================== 图像处理模块（模块化：独立封装图像相关逻辑） =====================
def init_tracking_client() -> bool:
    """
    初始化ros_api跟踪数据客户端（模块化：封装客户端初始化逻辑）
    :return: 初始化成功返回True，失败返回False
    """
    global tracking_client
    try:
        # 实例化ros_api客户端
        tracking_client = ros_api.TrackingDataClient(
            server_ip=ROS_SERVER_IP,
            port=ROS_SERVER_PORT
        )
        # 连接到ROS服务器
        return tracking_client.connect_to_server()
    except Exception as e:
        print(f"[{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}] ❌ 跟踪客户端初始化失败: {str(e)}")
        return False

def get_image_from_ros(timestamp: str) -> tuple[Image.Image, str] | tuple[None, str]:
    """
    从ros_api获取图像并转换为PIL Image（模块化：封装ROS图像获取逻辑）
    :param timestamp: 时间戳（用于生成图像文件名）
    :return: (PIL图像对象, 图像保存路径) 或 (None, 错误信息)
    """
    global tracking_client
    if not tracking_client:
        return None, "跟踪客户端未初始化"
    
    try:
        # 1. 发送请求到ROS服务器
        if not tracking_client.send_request():
            return None, "发送请求到ROS服务器失败"
        
        # 2. 解析字节流数据
        parsed_data = tracking_client.parse_byte_stream()
        if not parsed_data:
            return None, "解析ROS字节流数据失败"
        
        # 3. 提取OpenCV图像
        cv_image = parsed_data.get("current_image")
        if cv_image is None or not isinstance(cv_image, np.ndarray):
            return None, "从ROS数据中提取图像失败"
        
        # 4. OpenCV图像转换为PIL Image（CV2: BGR → PIL: RGB）
        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv_image_rgb)
        
        # 5. 保存图像到本地
        image_save_path = f"{IMAGE_SAVE_PREFIX}{timestamp}.jpg"
        pil_image.save(image_save_path)
        print(f"[{timestamp}] 💾 保存处理后图像到: {image_save_path}（尺寸：{pil_image.size}，模式：{pil_image.mode}）")
        
        return pil_image, image_save_path
    except Exception as e:
        error_msg = f"从ROS获取图像失败: {str(e)}"
        return None, error_msg

# ===================== 模型加载模块（模块化：独立封装模型相关逻辑） =====================
def load_model_vlm(model_path: str = MODEL_VLM_DEFAULT):
    """
    加载并编译AI模型（模块化：封装模型加载、编译逻辑）
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

def load_model_da3(model_path = MODEL_DA3_DEFAULT):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_da3 = DepthAnything3.from_pretrained(model_path)
    model_da3 = model_da3.to(device)
    model_da3.eval()
    print(f"{time.time()} > ✅ DA3 模型加载并编译完成！", flush=True)

def load_model_sam3(model_path = MODEL_SAM3_PATH):
    model = build_sam3_image_model(
        load_from_HF=False,
        checkpoint_path=model_path
    )
    processor_sam3 = Sam3Processor(model)

# ===================== 文本生成模块（模块化：独立封装流式生成逻辑） =====================
def generate_text_stream(text_query: str, image_path: str, timestamp: str):
    """
    流式生成文本响应（模块化：封装模型推理、流式返回逻辑）
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

# ===================== Flask接口模块（模块化：独立封装API逻辑） =====================
def init_flask_app():
    """
    初始化Flask应用（模块化：封装Flask配置、路由注册）
    """
    global app
    app = Flask(__name__)
    CORS(app)  # 允许跨域请求
    
    # 注册路由
    @app.route('/generate', methods=['POST'])
    def generate():
        # 1. 解析请求参数
        data = request.json or {}
        text_query = data.get('text', 'Describe this image')
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # 2. 从ROS获取图像
            print(f"[{timestamp}] 🔍 开始从ROS获取图像...")
            pil_image, image_path = get_image_from_ros(timestamp)
            if not pil_image:
                raise Exception(image_path)  # image_path此时为错误信息
            
            # 3. 流式返回生成结果
            return Response(generate_text_stream(text_query, image_path, timestamp), 
                            mimetype='text/event-stream')
        
        except Exception as e:
            error_msg = f"[{timestamp}] ❌ 错误: {str(e)}"
            print(error_msg)
            return jsonify({'error': error_msg}), 500

# ===================== 主程序入口（模块化：统一协调各模块初始化与运行） =====================
def main():
    """主程序（模块化：协调各模块初始化，启动服务）"""
    try:
        # 1. 初始化Flask应用
        init_flask_app()
        
        # 2. 初始化ROS跟踪客户端
        if not init_tracking_client():
            raise Exception("ROS跟踪客户端初始化失败，无法继续运行")
        
        # 3. 加载AI模型
        load_model_vlm(MODEL_VLM_DEFAULT)
        
        # 4. 启动Flask服务
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
    finally:
        # 收尾：关闭ROS客户端连接
        global tracking_client
        if tracking_client:
            tracking_client.close_connection()

if __name__ == '__main__':
    main()