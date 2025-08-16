"""
统一的LLM客户端，使用OpenAI兼容的API调用格式
支持多种模型提供商，包括OpenAI、硅基流动、本地VLLM等
"""

import os
import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from openai import OpenAI
from config import get_model_config, MODEL_CONFIGS

# 尝试导入backtest模块，如果不存在则使用简单的成本记录
try:
    from backtest.toolkit.llm_cost_monitor import add_llm_cost
    BACKTEST_AVAILABLE = True
except ImportError:
    BACKTEST_AVAILABLE = False
    _total_cost = 0.0
    def add_llm_cost(cost: float):
        global _total_cost
        _total_cost += cost

logger = logging.getLogger(__name__)

class LLMClientError(Exception):
    """LLM客户端异常"""
    pass

class UnifiedLLMClient:
    """统一的LLM客户端类"""
    
    def __init__(self, model_name: str):
        """
        初始化LLM客户端
        
        Args:
            model_name: 模型名称，必须在config.py的MODEL_CONFIGS中定义
        """
        self.model_name = model_name
        self.config = get_model_config(model_name)
        
        if self.config["type"] != "llm_api":
            raise ValueError(f"模型 {model_name} 不是LLM API类型")
            
        # 创建OpenAI客户端
        self.client = OpenAI(
            api_key=self.config["api_key"],
            base_url=self.config["api_base"]
        )
        
        # 设置重试参数
        self.max_retries = 3
        self.retry_delay = 1.0
        
        logger.info(f"初始化LLM客户端: {model_name} (提供商: {self.config['provider']})")
        
    def _retry_with_backoff(self, func, max_tries=3, base_delay=1.0, max_delay=30.0):
        """简单的指数退避重试机制"""
        for attempt in range(max_tries):
            try:
                return func()
            except Exception as e:
                if attempt == max_tries - 1:  # 最后一次尝试
                    raise e
                
                # 计算退避延迟
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning(f"API调用失败，{delay:.1f}秒后重试 (尝试 {attempt + 1}/{max_tries}): {e}")
                time.sleep(delay)
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: Optional[int] = 2000,
        top_p: float = 0.9,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        创建聊天完成
        
        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            temperature: 温度参数
            max_tokens: 最大token数
            top_p: top_p参数
            frequency_penalty: 频率惩罚
            presence_penalty: 存在惩罚
            **kwargs: 其他参数
            
        Returns:
            Tuple[str, Dict]: (响应内容, 使用信息)
        """
        def _api_call():
            response = self.client.chat.completions.create(
                model=self.config["model"],
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                **kwargs
            )
            return response
        
        try:
            response = self._retry_with_backoff(_api_call)
            
            content = response.choices[0].message.content
            
            # 提取使用信息
            usage_info = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
                "model": self.model_name,
                "provider": self.config["provider"]
            }
            
            # 记录成本
            self._record_cost(usage_info)
            
            return content, usage_info
            
        except Exception as e:
            logger.error(f"LLM API调用失败: {e}")
            raise LLMClientError(f"模型 {self.model_name} 调用失败: {str(e)}")
    
    def _record_cost(self, usage_info: Dict[str, Any]):
        """记录API调用成本"""
        try:
            # 估算成本（基于token数量，这里使用简化的成本模型）
            cost_per_1k_input = self._get_cost_per_1k_tokens("input")
            cost_per_1k_output = self._get_cost_per_1k_tokens("output")
            
            input_cost = (usage_info["prompt_tokens"] / 1000) * cost_per_1k_input
            output_cost = (usage_info["completion_tokens"] / 1000) * cost_per_1k_output
            total_cost = input_cost + output_cost
            
            add_llm_cost(total_cost)
            
            logger.debug(f"记录成本: ${total_cost:.6f} (输入: {usage_info['prompt_tokens']} tokens, "
                        f"输出: {usage_info['completion_tokens']} tokens)")
                        
        except Exception as e:
            logger.warning(f"记录成本失败: {e}")
    
    def _get_cost_per_1k_tokens(self, token_type: str) -> float:
        """获取每1000个token的成本"""
        # 这里是简化的成本模型，实际使用中可以根据具体模型配置
        provider = self.config["provider"]
        
        if provider == "siliconflow":
            # 硅基流动的大概成本
            return 0.001 if token_type == "input" else 0.002
        elif provider == "vllm":
            # 本地部署无成本
            return 0.0
        else:
            # 默认成本
            return 0.002 if token_type == "input" else 0.004
    
    def simple_completion(
        self,
        prompt: str,
        system_message: str = "You are a helpful assistant.",
        **kwargs
    ) -> str:
        """
        简单的文本完成
        
        Args:
            prompt: 用户输入
            system_message: 系统消息
            **kwargs: 其他参数
            
        Returns:
            str: 模型响应
        """
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        response, _ = self.chat_completion(messages, **kwargs)
        return response

class UnifiedEmbeddingClient:
    """统一的Embedding客户端类"""
    
    def __init__(self, model_name: str):
        """
        初始化Embedding客户端
        
        Args:
            model_name: 模型名称，必须在config.py的MODEL_CONFIGS中定义
        """
        self.model_name = model_name
        self.config = get_model_config(model_name)
        
        if self.config["type"] != "embedding_api":
            raise ValueError(f"模型 {model_name} 不是Embedding API类型")
            
        # 创建OpenAI客户端
        self.client = OpenAI(
            api_key=self.config["api_key"],
            base_url=self.config["api_base"]
        )
        
        logger.info(f"初始化Embedding客户端: {model_name} (提供商: {self.config['provider']})")
    
    def create_embeddings(
        self,
        texts: Union[str, List[str]]
    ) -> List[List[float]]:
        """
        创建文本嵌入
        
        Args:
            texts: 单个文本或文本列表
            
        Returns:
            List[List[float]]: 嵌入向量列表
        """
        if isinstance(texts, str):
            texts = [texts]
            
        def _embedding_call():
            response = self.client.embeddings.create(
                model=self.config["model"],
                input=texts
            )
            return response
        
        try:
            # 创建简单的重试机制
            for attempt in range(3):
                try:
                    response = _embedding_call()
                    break
                except Exception as e:
                    if attempt == 2:  # 最后一次尝试
                        raise e
                    time.sleep(1.0 * (2 ** attempt))
            
            embeddings = [item.embedding for item in response.data]
            
            logger.debug(f"创建了 {len(embeddings)} 个嵌入向量，维度: {len(embeddings[0])}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding API调用失败: {e}")
            raise LLMClientError(f"模型 {self.model_name} 嵌入创建失败: {str(e)}")
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询文本"""
        return self.create_embeddings([text])[0]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入多个文档文本"""
        return self.create_embeddings(texts)

# 便利函数
def create_llm_client(model_name: str) -> UnifiedLLMClient:
    """创建LLM客户端的便利函数"""
    return UnifiedLLMClient(model_name)

def create_embedding_client(model_name: str) -> UnifiedEmbeddingClient:
    """创建Embedding客户端的便利函数"""
    return UnifiedEmbeddingClient(model_name)

def get_available_models() -> Dict[str, List[str]]:
    """获取所有可用模型"""
    from config import get_all_llm_models, get_all_embedding_models
    
    return {
        "llm_models": get_all_llm_models(),
        "embedding_models": get_all_embedding_models()
    }

# 测试函数
def test_llm_client(model_name: str = "Qwen/Qwen3-8B"):
    """测试LLM客户端"""
    print(f"测试LLM客户端: {model_name}")
    
    try:
        client = create_llm_client(model_name)
        
        # 简单测试
        response = client.simple_completion("你好，请介绍一下你自己。")
        print(f"响应: {response[:200]}...")
        
        # 聊天测试
        messages = [
            {"role": "system", "content": "你是一个专业的金融分析师。"},
            {"role": "user", "content": "请分析一下当前的股票市场趋势。"}
        ]
        response, usage = client.chat_completion(messages, temperature=0.2, max_tokens=500)
        print(f"聊天响应: {response[:200]}...")
        print(f"使用信息: {usage}")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        return False

if __name__ == "__main__":
    # 运行测试
    print("=== 可用模型 ===")
    models = get_available_models()
    print(f"LLM模型: {models['llm_models']}")
    print(f"Embedding模型: {models['embedding_models']}")
    
    print("\n=== 测试LLM客户端 ===")
    test_llm_client()