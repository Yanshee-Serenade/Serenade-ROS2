from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import requests
import torch
import json
import datetime
import os
import time
from threading import Thread

MODEL_QWEN_8B="Qwen/Qwen3-VL-8B-Instruct"
MODEL_QWEN_4B="Qwen/Qwen3-VL-4B-Instruct"
MODEL_QWEN_2B="Qwen/Qwen3-VL-2B-Instruct"
MODEL_SMOLVLM="HuggingFaceTB/SmolVLM2-2.2B-Instruct"

app = Flask(__name__)
CORS(app)

docker_ip = "localhost"
port = 51121

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    text_query = data.get('text', 'Describe this image')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # 获取图像
        print(f"[{timestamp}] 🔍 开始获取图像...")
        image_response = requests.get(f"http://{docker_ip}:{port}/image")
        image_response.raise_for_status()
        print(f"[{timestamp}] ✅ 成功获取图像，大小: {len(image_response.content)} bytes")
        
        # 处理图像
        pil_image = Image.open(BytesIO(image_response.content))
        print(f"[{timestamp}] 🖼️ 图像尺寸: {pil_image.size}, 模式: {pil_image.mode}")
        
        # 保存处理后的图像
        processed_path = f"image_{timestamp}.jpg"
        pil_image.save(processed_path)
        print(f"[{timestamp}] 💾 保存处理后图像到: {processed_path}")
        
        def generate_stream():
            print(f"[{timestamp}] 🤖 开始生成文本，查询: '{text_query}'")
            
            # 构建消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "path": processed_path},
                        {"type": "text", "text": text_query}
                    ]
                },
            ]
            
            # 应用模板
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device, dtype=torch.bfloat16)
            
            # 创建流式生成器
            streamer = TextIteratorStreamer(
                processor.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            # 在单独的线程中运行生成
            generation_kwargs = dict(
                inputs,
                streamer=streamer,
                max_new_tokens=256,
                do_sample=True,
                num_beams=1
            )
            
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # 从streamer中获取生成的文本并发送
            for new_text in streamer:
                if new_text:
                    # print(f"[{timestamp}] 📝 新文本: {new_text}")
                    yield f"data: {json.dumps({'text': new_text})}\n\n"
            
            print(f"[{timestamp}] ✅ 生成完成")
            
        return Response(generate_stream(), mimetype='text/event-stream')
        
    except Exception as e:
        error_msg = f"[{timestamp}] ❌ 错误: {str(e)}"
        print(error_msg)
        return jsonify({'error': error_msg}), 500


if __name__ == '__main__':
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from transformers import TextIteratorStreamer
    
    model_path = MODEL_QWEN_4B
    processor = AutoProcessor.from_pretrained(model_path)

    print(f"{time.time()} > Loading model...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        # dtype=torch.bfloat16,
        load_in_8bit=True
    )

    print(f"{time.time()} > Compiling model...", flush=True)
    model = torch.compile(model)

    print(f"{time.time()} > Model is compiled!", flush=True)
    app.run(host='0.0.0.0', port=51122, threaded=True)