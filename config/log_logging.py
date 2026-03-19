import logging
import json
import time
import uuid
import os
from functools import wraps


os.makedirs("data", exist_ok=True)

# 初始化专门给 LLM 用的 Logger
llm_logger = logging.getLogger("LLM_TRACKER")
llm_logger.setLevel(logging.INFO)

# 强制使用 JSON 格式，方便后期导入 ElasticSearch 或其他分析工具
json_handler = logging.FileHandler("data/llm_trace.log", encoding="utf-8")
json_handler.setFormatter(logging.Formatter('%(message)s')) 
llm_logger.addHandler(json_handler)

def track_llm_call(model_name="unknown"):
    """
    LLM 调用追踪装饰器
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            trace_id = str(uuid.uuid4()) # 生成唯一的请求ID
            start_time = time.time()
            
            # 提取传入的提示词 (假设你的函数接收 prompt 参数)
            prompt = kwargs.get('prompt', args[0] if args else "")
            
            try:
                # 执行真正的 LLM 调用
                response = func(*args, **kwargs)
                latency = round(time.time() - start_time, 2)
                
                # 假设 response 是 OpenAI SDK 的返回格式
                # 实际业务中你需要根据你用的框架(Langchain/原版 SDK)调整提取逻辑
                completion_text = response.choices[0].message.content
                usage = response.usage
                
                log_data = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "trace_id": trace_id,
                    "model": model_name,
                    "status": "success",
                    "latency_sec": latency,
                    "usage": {
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens
                    },
                    "prompt": prompt,
                    "response": completion_text
                }
                
                # 记录 JSON 日志
                llm_logger.info(json.dumps(log_data, ensure_ascii=False))
                return response
                
            except Exception as e:
                latency = round(time.time() - start_time, 2)
                error_log = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "trace_id": trace_id,
                    "model": model_name,
                    "status": "error",
                    "latency_sec": latency,
                    "error_msg": str(e),
                    "prompt": prompt
                }
                llm_logger.error(json.dumps(error_log, ensure_ascii=False))
                raise e # 继续抛出异常，不吞掉报错
                
        return wrapper
    return decorator