"""
统一的聊天接口，替代原有的chat.py
使用新的LLM客户端进行API调用
"""

import logging
from typing import Dict, Any, List, Union, Optional
from llm_client import create_llm_client, LLMClientError

logger = logging.getLogger(__name__)

class UnifiedChatClient:
    """统一的聊天客户端，使用新的LLM API"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        system_message: str = "You are a helpful assistant.",
        **default_params
    ):
        """
        初始化聊天客户端
        
        Args:
            model_name: 模型名称，必须在config.py中配置
            system_message: 系统消息
            **default_params: 默认参数
        """
        self.model_name = model_name
        self.system_message = system_message
        self.llm_client = create_llm_client(model_name)
        
        # 默认参数
        self.default_params = {
            "temperature": 0.1,
            "max_tokens": 2000,
            "top_p": 0.9,
            **default_params
        }
        
        logger.info(f"初始化聊天客户端，模型: {model_name}")
    
    def chat(
        self,
        user_input: str,
        system_message: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> tuple[str, Dict[str, Any]]:
        """
        进行聊天对话
        
        Args:
            user_input: 用户输入
            system_message: 系统消息（可选，覆盖默认系统消息）
            conversation_history: 对话历史
            **kwargs: 其他参数
            
        Returns:
            Tuple[str, Dict]: (响应内容, 使用信息)
        """
        # 构建消息列表
        messages = []
        
        # 添加系统消息
        sys_msg = system_message or self.system_message
        if sys_msg:
            messages.append({"role": "system", "content": sys_msg})
        
        # 添加对话历史
        if conversation_history:
            messages.extend(conversation_history)
        
        # 添加当前用户输入
        messages.append({"role": "user", "content": user_input})
        
        # 合并参数
        params = {**self.default_params, **kwargs}
        
        try:
            response, usage = self.llm_client.chat_completion(messages, **params)
            
            logger.debug(f"聊天完成，使用tokens: {usage.get('total_tokens', 0)}")
            
            return response, usage
            
        except LLMClientError as e:
            logger.error(f"聊天API调用失败: {e}")
            raise
    
    def simple_chat(self, user_input: str, **kwargs) -> str:
        """简单聊天，只返回响应内容"""
        response, _ = self.chat(user_input, **kwargs)
        return response
    
    def guardrail_chat(
        self,
        user_input: str,
        require_json: bool = False,
        **kwargs
    ) -> str:
        """
        带防护栏的聊天，确保输出格式
        
        Args:
            user_input: 用户输入
            require_json: 是否要求JSON格式输出
            **kwargs: 其他参数
            
        Returns:
            str: 响应内容
        """
        if require_json:
            system_msg = "You are a helpful assistant only capable of communicating with valid JSON, and no other text."
        else:
            system_msg = self.system_message
        
        response, _ = self.chat(
            user_input,
            system_message=system_msg,
            temperature=0.0,  # 降低温度以获得更一致的输出
            **kwargs
        )
        
        return response

# 向后兼容的类，替代原有的ChatOpenAICompatible
class ChatOpenAICompatible:
    """向后兼容的聊天类"""
    
    def __init__(
        self,
        end_point: str = None,  # 现在不再使用，保留以兼容
        model: str = "Qwen/Qwen3-8B",
        system_message: str = "You are a helpful assistant.",
        other_parameters: Union[Dict[str, Any], None] = None,
    ):
        """
        初始化兼容客户端
        
        Args:
            end_point: 端点URL（已弃用，保留兼容性）
            model: 模型名称
            system_message: 系统消息
            other_parameters: 其他参数
        """
        self.model = model
        self.system_message = system_message
        self.other_parameters = other_parameters or {}
        
        # 创建统一聊天客户端
        self.unified_client = UnifiedChatClient(
            model_name=model,
            system_message=system_message,
            **self.other_parameters
        )
        
        logger.info(f"创建兼容聊天客户端，模型: {model}")
    
    def guardrail_endpoint(self):
        """返回防护栏端点函数，保持API兼容性"""
        
        def endpoint_func(input_text: str, **kwargs) -> str:
            """端点函数"""
            return self.unified_client.guardrail_chat(
                input_text,
                require_json=True,
                **kwargs
            )
        
        return endpoint_func
    
    def parse_response(self, response) -> str:
        """解析响应（已弃用，保留兼容性）"""
        # 新的客户端直接返回字符串，无需解析
        return str(response)

# 便利函数
def create_chat_client(
    model_name: str = "Qwen/Qwen3-8B",
    system_message: str = "You are a helpful assistant.",
    **params
) -> UnifiedChatClient:
    """创建聊天客户端的便利函数"""
    return UnifiedChatClient(model_name, system_message, **params)

def quick_chat(
    prompt: str,
    model_name: str = "Qwen/Qwen3-8B",
    system_message: str = "You are a helpful assistant.",
    **params
) -> str:
    """快速聊天函数"""
    client = create_chat_client(model_name, system_message, **params)
    return client.simple_chat(prompt)

# 异常类（向后兼容）
class LongerThanContextError(Exception):
    """上下文长度超限异常"""
    pass