"""
LLM客户端封装
支持 OpenAI 格式、Vertex AI Token 自动刷新及多模型切换
"""

import json
import re
import os
import subprocess
import time
from typing import Optional, Dict, Any, List
from openai import OpenAI

from ..config import Config
from .logger import get_logger

logger = get_logger('mirofish.llm_client')

class LLMClient:
    """
    LLM客户端
    
    支持:
    1. 标准 OpenAI API
    2. Vertex AI OpenAI-compatible endpoint (带 Token 自动刷新)
    3. 多模型配置切换
    """
    
    # 全局 Token 缓存，避免频繁调用 gcloud
    _token_cache = {
        "token": None,
        "expiry": 0
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        初始化客户端
        
        Args:
            api_key: API Key (可选，默认从 Config 读取)
            base_url: Base URL (可选，默认从 Config 读取)
            model: 模型名称或标签 (如 'gemini-pro', 'opus')
        """
        # 处理模型标签切换
        if model in Config.MODELS_CONFIG:
            model_cfg = Config.MODELS_CONFIG[model]
            self.model = model_cfg["model"]
            self.base_url = model_cfg.get("base_url") or Config.LLM_BASE_URL
            self.api_key = model_cfg.get("api_key") or api_key or Config.LLM_API_KEY
        else:
            self.model = model or Config.LLM_MODEL_NAME
            self.base_url = base_url or Config.LLM_BASE_URL
            self.api_key = api_key or Config.LLM_API_KEY
        
        # 记录原始 API Key，用于判断是否需要刷新 (Vertex AI 环境下通常初始也是 Token)
        self.is_vertex = 'googleapis.com' in self.base_url
        
        if not self.api_key and not self.is_vertex:
            raise ValueError("LLM_API_KEY 未配置且非 Vertex AI 环境")
        
        self._init_client()

    def _init_client(self):
        """初始化 OpenAI 客户端实例"""
        current_api_key = self.api_key
        
        if self.is_vertex:
            current_api_key = self._get_vertex_token()
            
        self.client = OpenAI(
            api_key=current_api_key,
            base_url=self.base_url
        )

    def _get_vertex_token(self) -> str:
        """获取并缓存 Vertex AI Access Token"""
        now = time.time()
        # 如果缓存有效 (提前 5 分钟刷新)，直接返回
        if self._token_cache["token"] and now < self._token_cache["expiry"] - 300:
            return self._token_cache["token"]
            
        try:
            logger.info("正在通过 gcloud 刷新 Vertex AI Access Token...")
            result = subprocess.run(
                ['gcloud', 'auth', 'print-access-token'],
                capture_output=True,
                text=True,
                check=True,
                shell=True if os.name == 'nt' else False
            )
            token = result.stdout.strip()
            if not token:
                raise ValueError("gcloud 返回的 token 为空")
                
            self._token_cache["token"] = token
            self._token_cache["expiry"] = now + 3600 # 默认 1 小时有效期
            return token
        except Exception as e:
            logger.error(f"刷新 Vertex AI Token 失败: {e}")
            # 如果刷新失败但配置了初始 API Key，则回退使用初始值
            if self.api_key:
                return self.api_key
            raise

    def _ensure_active_token(self):
        """确保 Token 有效，如果失效则重新初始化客户端"""
        if self.is_vertex:
            now = time.time()
            if now >= self._token_cache["expiry"] - 300:
                self._init_client()

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None
    ) -> str:
        """发送聊天请求"""
        self._ensure_active_token()
        
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            # "max_tokens": max_tokens, # 部分模型对 max_tokens 敏感，可选
        }
        
        # 某些模型可能不支持 max_tokens 参数或有不同限制
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
            
        if response_format:
            kwargs["response_format"] = response_format
        
        try:
            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            # 部分模型（如MiniMax M2.5）会在content中包含<think>思考内容，需要移除
            content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
            return content
        except Exception as e:
            logger.error(f"LLM 请求失败: {e}")
            # 如果是认证错误且是 Vertex AI，尝试刷新一次再试
            if self.is_vertex and ("401" in str(e) or "authentication" in str(e).lower()):
                logger.info("认证失败，尝试强制刷新 Token 并重试...")
                self._token_cache["expiry"] = 0
                self._init_client()
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
            raise

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """发送聊天请求并返回JSON"""
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        # 清理markdown代码块标记
        cleaned_response = response.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            # 尝试修复常见的 JSON 错误，如尾部逗号
            try:
                # 简单替换一些明显的错误
                fixed = re.sub(r',\s*}', '}', cleaned_response)
                fixed = re.sub(r',\s*]', ']', fixed)
                return json.loads(fixed)
            except:
                raise ValueError(f"LLM返回的JSON格式无效: {cleaned_response}")
