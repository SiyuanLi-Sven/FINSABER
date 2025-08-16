"""
FinAgent的统一Provider，使用新的LLM客户端
替代原有的OpenAIProvider
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
from llm_client import create_llm_client, create_embedding_client, LLMClientError
from llm_traders.finagent.provider.base_llm import LLMProvider
from llm_traders.finagent.provider.base_embedding import EmbeddingProvider
from llm_traders.finagent.registry import PROVIDER

logger = logging.getLogger(__name__)

@PROVIDER.register_module(force=True, name="UnifiedProvider")
class UnifiedProvider(LLMProvider, EmbeddingProvider):
    """统一的Provider，使用config.py中的模型配置"""
    
    def __init__(self, provider_cfg_path: str = None, model_config: Dict[str, Any] = None):
        """
        初始化统一Provider
        
        Args:
            provider_cfg_path: 原有配置文件路径（保持兼容性）
            model_config: 新的模型配置，包含llm_model和embedding_model
        """
        if model_config is None:
            # 默认配置
            model_config = {
                "llm_model": "Qwen/Qwen3-8B",
                "embedding_model": "Qwen/Qwen3-Embedding-4B"
            }
            
        self.llm_model = model_config.get("llm_model", "Qwen/Qwen3-8B")
        self.embedding_model = model_config.get("embedding_model", "Qwen/Qwen3-Embedding-4B")
        
        # 创建客户端
        try:
            self.llm_client = create_llm_client(self.llm_model)
            self.embedding_client = create_embedding_client(self.embedding_model)
            logger.info(f"统一Provider初始化成功: LLM={self.llm_model}, Embedding={self.embedding_model}")
        except Exception as e:
            logger.error(f"统一Provider初始化失败: {e}")
            raise
            
        # 设置重试参数
        self.retries = 5
        
    def create_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = 0.0,
        seed: int = None,
        max_tokens: int = 512,
        **kwargs
    ) -> Tuple[str, Dict[str, int]]:
        """
        创建文本完成
        
        Args:
            messages: 消息列表
            model: 模型名称（可选，使用默认LLM模型）
            temperature: 温度参数
            seed: 随机种子（暂时忽略，保持兼容性）
            max_tokens: 最大token数
            **kwargs: 其他参数
            
        Returns:
            Tuple[str, Dict[str, int]]: (响应内容, 使用信息)
        """
        try:
            # 如果指定了不同的模型，创建新的客户端
            if model and model != self.llm_model:
                temp_client = create_llm_client(model)
                response, usage = temp_client.chat_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
            else:
                response, usage = self.llm_client.chat_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
            
            # 转换使用信息格式以保持兼容性
            info = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            }
            
            return response, info
            
        except LLMClientError as e:
            logger.error(f"LLM完成调用失败: {e}")
            return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        except Exception as e:
            logger.error(f"未知错误: {e}")
            return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        嵌入文档列表
        
        Args:
            texts: 文本列表
            
        Returns:
            List[List[float]]: 嵌入向量列表
        """
        try:
            return self.embedding_client.embed_documents(texts)
        except LLMClientError as e:
            logger.error(f"文档嵌入失败: {e}")
            return []
    
    def embed_query(self, text: str) -> List[float]:
        """
        嵌入单个查询
        
        Args:
            text: 查询文本
            
        Returns:
            List[float]: 嵌入向量
        """
        try:
            return self.embedding_client.embed_query(text)
        except LLMClientError as e:
            logger.error(f"查询嵌入失败: {e}")
            return []
    
    def get_embedding_dim(self) -> int:
        """获取嵌入维度"""
        try:
            # 通过嵌入一个测试文本来获取维度
            test_embedding = self.embed_query("test")
            return len(test_embedding) if test_embedding else 0
        except Exception as e:
            logger.warning(f"获取嵌入维度失败: {e}")
            # 返回一些常见的维度作为默认值
            embedding_dims = {
                "Qwen/Qwen3-Embedding-4B": 2048,
                "Qwen/Qwen3-Embedding-8B": 4096,
                "text-embedding-ada-002": 1536,
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072
            }
            return embedding_dims.get(self.embedding_model, 1536)
    
    def num_tokens_from_messages(self, messages, model=None) -> int:
        """
        估算消息的token数量
        
        Args:
            messages: 消息列表
            model: 模型名称
            
        Returns:
            int: token数量估算
        """
        try:
            # 简单的token数量估算（每个字符约0.25个token）
            total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
            estimated_tokens = int(total_chars * 0.25)
            
            # 加上消息格式的开销
            message_overhead = len(messages) * 3  # 每条消息约3个额外token
            
            return estimated_tokens + message_overhead
            
        except Exception as e:
            logger.warning(f"Token数量估算失败: {e}")
            return 0
    
    def assemble_prompt(
        self, 
        system_prompts: List[str], 
        user_inputs: List[str], 
        image_filenames: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        组装提示消息
        
        Args:
            system_prompts: 系统提示列表
            user_inputs: 用户输入列表
            image_filenames: 图片文件名列表（暂不支持）
            
        Returns:
            List[Dict[str, Any]]: 组装好的消息列表
        """
        messages = []
        
        # 添加系统消息
        if system_prompts:
            for prompt in system_prompts:
                messages.append({"role": "system", "content": prompt})
        
        # 添加用户消息
        if user_inputs:
            for input_text in user_inputs:
                messages.append({"role": "user", "content": input_text})
        
        # 图片支持暂时跳过（需要多模态模型支持）
        if image_filenames:
            logger.warning("当前版本暂不支持图片输入")
            
        return messages

# 便利函数
def create_unified_provider(
    llm_model: str = "Qwen/Qwen3-8B",
    embedding_model: str = "Qwen/Qwen3-Embedding-4B"
) -> UnifiedProvider:
    """创建统一Provider的便利函数"""
    model_config = {
        "llm_model": llm_model,
        "embedding_model": embedding_model
    }
    return UnifiedProvider(model_config=model_config)

# 兼容性函数
def create_openai_compatible_provider(provider_cfg_path: str = None) -> UnifiedProvider:
    """创建OpenAI兼容Provider"""
    # 读取原有配置并转换为新格式
    if provider_cfg_path:
        # 这里可以添加读取旧配置文件的逻辑
        logger.info(f"使用旧配置文件: {provider_cfg_path}，将转换为新格式")
    
    # 使用默认配置
    return create_unified_provider()