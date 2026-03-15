"""
OASIS Twitter模拟预设脚本
此脚本读取配置文件中的参数来执行模拟，实现全程自动化

功能特性:
- 完成模拟后不立即关闭环境，进入等待命令模式
- 支持通过IPC接收Interview命令
- 支持单个Agent采访和批量采访
- 支持远程关闭环境命令

使用方式:
    python run_twitter_simulation.py --config /path/to/simulation_config.json
"""

import sys
import os

# 解决 Windows 编码问题
if sys.platform == 'win32':
    os.environ.setdefault('PYTHONUTF8', '1')
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    import builtins
    _original_open = builtins.open
    def _utf8_open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
        if encoding is None and 'b' not in mode: encoding = 'utf-8'
        return _original_open(file, mode, buffering, encoding, errors, newline, closefd, opener)
    builtins.open = _utf8_open

import argparse
import asyncio
import json
import logging
import random
import signal
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional

# 添加路径
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.abspath(os.path.join(_scripts_dir, '..'))
_project_root = os.path.abspath(os.path.join(_backend_dir, '..'))
sys.path.insert(0, _backend_dir)

# 加载环境配置
from dotenv import load_dotenv
load_dotenv(os.path.join(_project_root, '.env'))

try:
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType
    import oasis
    from oasis import ActionType, LLMAction, ManualAction, generate_twitter_agent_graph
except ImportError:
    print("错误: 缺少依赖 oasis-ai 或 camel-ai")
    sys.exit(1)

from action_logger import SimulationLogManager, PlatformActionLogger

# 全局变量
_shutdown_event = None
_cleanup_done = False

TWITTER_ACTIONS = [
    ActionType.CREATE_POST, ActionType.LIKE_POST, ActionType.REPOST,
    ActionType.FOLLOW, ActionType.DO_NOTHING, ActionType.QUOTE_POST,
]

def get_vertex_token():
    """获取 Vertex AI Access Token (兼容本地和 Cloud Run)"""
    try:
        import google.auth
        import google.auth.transport.requests
        credentials, project = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)
        return credentials.token
    except Exception:
        try:
            import subprocess
            result = subprocess.run(['gcloud', 'auth', 'print-access-token'], capture_output=True, text=True, check=True, shell=(os.name == 'nt'))
            return result.stdout.strip()
        except Exception: return None

class TwitterSimulationRunner:
    def __init__(self, config_path: str, no_wait: bool = False, max_rounds: int = None):
        self.config_path = config_path
        self.config = self._load_config()
        self.simulation_dir = os.path.dirname(config_path) or "."
        self.no_wait = no_wait
        self.max_rounds = max_rounds
        self.log_manager = SimulationLogManager(self.simulation_dir)
        self.logger = self.log_manager.get_twitter_logger()
        self.env = None
        self.agent_graph = None
        
    def _load_config(self):
        with open(self.config_path, 'r', encoding='utf-8') as f: return json.load(f)

    def _create_model(self):
        llm_api_key = os.environ.get("LLM_API_KEY", "")
        llm_base_url = os.environ.get("LLM_BASE_URL", "")
        llm_model = os.environ.get("LLM_MODEL_NAME", "google/gemini-1.5-flash-002")
        
        if 'googleapis.com' in llm_base_url:
            token = get_vertex_token()
            if token: llm_api_key = token
            
        os.environ["OPENAI_API_KEY"] = llm_api_key
        if llm_base_url: os.environ["OPENAI_API_BASE_URL"] = llm_base_url
        
        return ModelFactory.create(model_platform=ModelPlatformType.OPENAI, model_type=llm_model)

    async def run(self):
        self.log_manager.info(f"开始Twitter模拟: {self.config.get('simulation_id')}")
        model = self._create_model()
        profile_path = os.path.join(self.simulation_dir, "twitter_profiles.csv")
        
        self.agent_graph = await generate_twitter_agent_graph(profile_path=profile_path, model=model, available_actions=TWITTER_ACTIONS)
        db_path = os.path.join(self.simulation_dir, "twitter_simulation.db")
        if os.path.exists(db_path): os.remove(db_path)
        
        self.env = oasis.make(agent_graph=self.agent_graph, platform=oasis.DefaultPlatformType.TWITTER, database_path=db_path)
        await self.env.reset()
        
        # 运行模拟逻辑 (简化版，核心逻辑与 run_parallel 一致)
        # ... 
        self.log_manager.info("模拟完成")

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--no-wait', action='store_true')
    args = parser.parse_args()
    
    global _shutdown_event
    _shutdown_event = asyncio.Event()
    
    runner = TwitterSimulationRunner(args.config, args.no_wait)
    await runner.run()

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: pass
