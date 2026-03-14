"""
配置管理
统一从项目根目录的 .env 文件加载配置
"""

import os
from dotenv import load_dotenv

# 加载项目根目录的 .env 文件
# 路径: MiroFish/.env (相对于 backend/app/config.py)
project_root_env = os.path.join(os.path.dirname(__file__), '../../.env')

if os.path.exists(project_root_env):
    load_dotenv(project_root_env, override=True)
else:
    # 如果根目录没有 .env，尝试加载环境变量（用于生产环境）
    load_dotenv(override=True)


class Config:
    """Flask配置类"""
    
    # Flask配置
    SECRET_KEY = os.environ.get('SECRET_KEY', 'mirofish-secret-key')
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    # JSON配置 - 禁用ASCII转义，让中文直接显示（而不是 \uXXXX 格式）
    JSON_AS_ASCII = False
    
    # LLM配置（Vertex AI OpenAI-compatible endpoint）
    LLM_API_KEY = os.environ.get('LLM_API_KEY')
    # Vertex AI OpenAI-compatible 基础 URL 格式 (注意末尾不带 /，openai 会自动加 /chat/completions)
    LLM_BASE_URL = os.environ.get('LLM_BASE_URL', 'https://us-central1-aiplatform.googleapis.com/v1/projects/reference-city-xrjsb/locations/us-central1/endpoints/openapi')
    LLM_MODEL_NAME = os.environ.get('LLM_MODEL_NAME', 'google/gemini-1.5-flash-002')
    
    # 扩展：支持多个 Vertex AI / OpenAI 模型
    MODELS_CONFIG = {
        "gemini-pro": {
            "model": os.environ.get('GEMINI_PRO_MODEL', 'google/gemini-1.5-pro-002'),
            "base_url": LLM_BASE_URL,
        },
        "gemini-flash": {
            "model": os.environ.get('GEMINI_FLASH_MODEL', 'google/gemini-1.5-flash-002'),
            "base_url": LLM_BASE_URL,
        },
        "opus": {
            "model": os.environ.get('OPUS_MODEL', 'google/claude-3-opus@20240229'),
            "base_url": LLM_BASE_URL,
        },
        "sonnet": {
            "model": os.environ.get('SONNET_MODEL', 'google/claude-3-5-sonnet@20240620'),
            "base_url": LLM_BASE_URL,
        }
    }

    # 是否是 Vertex AI 环境 (自动检测)
    IS_VERTEX_AI = 'googleapis.com' in LLM_BASE_URL
    
    # Zep配置
    ZEP_API_KEY = os.environ.get('ZEP_API_KEY')
    
    # 文件上传配置
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../uploads')
    ALLOWED_EXTENSIONS = {'pdf', 'md', 'txt', 'markdown'}
    
    # 文本处理配置
    DEFAULT_CHUNK_SIZE = 500  # 默认切块大小
    DEFAULT_CHUNK_OVERLAP = 50  # 默认重叠大小
    
    # OASIS模拟配置
    OASIS_DEFAULT_MAX_ROUNDS = int(os.environ.get('OASIS_DEFAULT_MAX_ROUNDS', '10'))
    OASIS_SIMULATION_DATA_DIR = os.path.join(os.path.dirname(__file__), '../uploads/simulations')
    
    # OASIS平台可用动作配置
    OASIS_TWITTER_ACTIONS = [
        'CREATE_POST', 'LIKE_POST', 'REPOST', 'FOLLOW', 'DO_NOTHING', 'QUOTE_POST'
    ]
    OASIS_REDDIT_ACTIONS = [
        'LIKE_POST', 'DISLIKE_POST', 'CREATE_POST', 'CREATE_COMMENT',
        'LIKE_COMMENT', 'DISLIKE_COMMENT', 'SEARCH_POSTS', 'SEARCH_USER',
        'TREND', 'REFRESH', 'DO_NOTHING', 'FOLLOW', 'MUTE'
    ]
    
    # Report Agent配置
    REPORT_AGENT_MAX_TOOL_CALLS = int(os.environ.get('REPORT_AGENT_MAX_TOOL_CALLS', '5'))
    REPORT_AGENT_MAX_REFLECTION_ROUNDS = int(os.environ.get('REPORT_AGENT_MAX_REFLECTION_ROUNDS', '2'))
    REPORT_AGENT_TEMPERATURE = float(os.environ.get('REPORT_AGENT_TEMPERATURE', '0.5'))
    
    @classmethod
    def validate(cls):
        """验证必要配置"""
        errors = []
        if not cls.LLM_API_KEY and not cls.IS_VERTEX_AI:
            # Vertex AI 环境下 LLM_API_KEY 是动态生成的，可以允许初始为空
            pass
        elif not cls.LLM_API_KEY and not cls.IS_VERTEX_AI:
            errors.append("LLM_API_KEY 未配置")
            
        # Zep 为可选配置，如果未配置，部分图谱记忆功能将不可用
        if not cls.ZEP_API_KEY:
            print("提示: ZEP_API_KEY 未配置，图谱记忆功能将被禁用")
            
        return errors

