import sys
import os

# 添加 backend 目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.utils.llm_client import LLMClient
from app.config import Config

def test_models():
    print("开始测试 LLMClient 多模型支持及 Vertex AI Token 自动刷新...")
    
    models_to_test = ["gemini-pro", "gemini-flash", "opus"]
    
    for model_label in models_to_test:
        print(f"\n--- 测试模型: {model_label} ---")
        try:
            client = LLMClient(model=model_label)
            print(f"初始化成功: model={client.model}, base_url={client.base_url}")
            
            response = client.chat([{"role": "user", "content": "你好，请简单介绍一下你自己。"}])
            print(f"响应成功:\n{response}")
            
        except Exception as e:
            print(f"测试失败: {e}")

if __name__ == "__main__":
    test_models()
