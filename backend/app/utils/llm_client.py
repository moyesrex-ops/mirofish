"""
LLM客户端封装
支持 OpenAI 格式、Vertex AI Token 自动刷新及多模型切换
"""

import json
import re
import os
import time
from typing import Optional, Dict, Any, List
from openai import OpenAI

# 导入 Google Auth 用于生产级 Token 获取
try:
    import google.auth
    import google.auth.transport.requests
    HAS_GOOGLE_AUTH = True
except ImportError:
    HAS_GOOGLE_AUTH = False

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
    
    # 全局 Token 缓存，避免频繁调用认证
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
        """
        if model in Config.MODELS_CONFIG:
            model_cfg = Config.MODELS_CONFIG[model]
            self.model = model_cfg["model"]
            self.base_url = model_cfg.get("base_url") or Config.LLM_BASE_URL
            self.api_key = model_cfg.get("api_key") or api_key or Config.LLM_API_KEY
        else:
            self.model = model or Config.LLM_MODEL_NAME
            self.base_url = base_url or Config.LLM_BASE_URL
            self.api_key = api_key or Config.LLM_API_KEY
        
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
        """获取并缓存 Vertex AI Access Token (兼容本地和 Cloud Run)"""
        now = time.time()
        # 如果缓存有效 (提前 5 分钟刷新)，直接返回
        if self._token_cache["token"] and now < self._token_cache["expiry"] - 300:
            return self._token_cache["token"]
            
        try:
            if HAS_GOOGLE_AUTH:
                logger.info("使用 google-auth 获取 Vertex AI Access Token...")
                credentials, project = google.auth.default(
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
                auth_req = google.auth.transport.requests.Request()
                credentials.refresh(auth_req)
                token = credentials.token
                # 获取 Token 有效期
                expiry = credentials.expiry.timestamp() if credentials.expiry else (now + 3600)
            else:
                # 备用：本地开发环境下如果没有安装 google-auth 且有 gcloud
                import subprocess
                logger.info("正在通过 gcloud 刷新 Vertex AI Access Token...")
                result = subprocess.run(
                    ['gcloud', 'auth', 'print-access-token'],
                    capture_output=True, text=True, check=True, shell=(os.name == 'nt')
                )
                token = result.stdout.strip()
                expiry = now + 3600
                
            if not token:
                raise ValueError("获取到的 Token 为空")
                
            self._token_cache["token"] = token
            self._token_cache["expiry"] = expiry
            return token
        except Exception as e:
            logger.error(f"刷新 Vertex AI Token 失败: {e}")
            if self.api_key and self.api_key != "initial_placeholder":
                return self.api_key
            raise

    def _ensure_active_token(self):
        """确保 Token 有效"""
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
        self._ensure_active_token()
        
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
            
        if response_format:
            kwargs["response_format"] = response_format
        
        try:
            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
            return content
        except Exception as e:
            logger.error(f"LLM 请求失败: {e}")
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
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        cleaned_response = response.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            try:
                fixed = re.sub(r',\s*}', '}', cleaned_response)
                fixed = re.sub(r',\s*]', ']', fixed)
                return json.loads(fixed)
            except:
                raise ValueError(f"LLM返回的JSON格式无效: {cleaned_response}")
