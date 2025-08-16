# 除非另有说明, 全部的服务都是openai sdk兼容的
# 包括llm服务和embedding服务
# 哪怕是本地部署的vllm, 也要走openai的sdk来进行

# 核心理念是每个模型都可以设置独立的api_base

# 模型配置 - 统一使用OpenAI兼容的API调用
MODEL_CONFIGS = {
    # === LLM 模型配置 ===
    
    # 硅基流动 API
    "Qwen/Qwen3-8B": {
        "type": "llm_api",
        "model": "Qwen/Qwen3-8B",
        "api_base": "https://api.siliconflow.cn/v1",
        "api_key": "sk-dovbcvocsibhdmuldjaflbyoqjrdukllzlcfzhkgsmjmvotn",
        "provider": "siliconflow"
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "type": "llm_api",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "api_base": "https://api.siliconflow.cn/v1", 
        "api_key": "sk-dovbcvocsibhdmuldjaflbyoqjrdukllzlcfzhkgsmjmvotn",
        "provider": "siliconflow"
    },
    "deepseek-ai/DeepSeek-V3": {
        "type": "llm_api",
        "model": "deepseek-ai/DeepSeek-V3",
        "api_base": "https://api.siliconflow.cn/v1", 
        "api_key": "sk-dovbcvocsibhdmuldjaflbyoqjrdukllzlcfzhkgsmjmvotn",
        "provider": "siliconflow"
    },
    "deepseek-ai/DeepSeek-R1": {
        "type": "llm_api",
        "model": "deepseek-ai/DeepSeek-R1",
        "api_base": "https://api.siliconflow.cn/v1", 
        "api_key": "sk-dovbcvocsibhdmuldjaflbyoqjrdukllzlcfzhkgsmjmvotn",
        "provider": "siliconflow"
    },
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "type": "llm_api",
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "api_base": "https://api.siliconflow.cn/v1",
        "api_key": "sk-dovbcvocsibhdmuldjaflbyoqjrdukllzlcfzhkgsmjmvotn", 
        "provider": "siliconflow"
    },
    
    # 本地部署的VLLM (通过OpenAI兼容接口)
    "qwen3_local-vllm": {
        "type": "llm_api",
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",  # 或其他本地模型
        "api_base": "http://0.0.0.0:8000/v1", 
        "api_key": "EMPTY",  # VLLM本地部署通常不需要key
        "provider": "vllm"
    },
    
    # === Embedding 模型配置 ===
    
    # 硅基流动 Embedding
    "Qwen/Qwen3-Embedding-4B": {
        "type": "embedding_api", 
        "model": "Qwen/Qwen3-Embedding-4B",
        "api_base": "https://api.siliconflow.cn/v1",
        "api_key": "sk-dovbcvocsibhdmuldjaflbyoqjrdukllzlcfzhkgsmjmvotn",
        "provider": "siliconflow",
        "dimensions": 2048
    },
    "Qwen/Qwen3-Embedding-8B": {
        "type": "embedding_api", 
        "model": "Qwen/Qwen3-Embedding-8B",
        "api_base": "https://api.siliconflow.cn/v1",
        "api_key": "sk-dovbcvocsibhdmuldjaflbyoqjrdukllzlcfzhkgsmjmvotn",
        "provider": "siliconflow",
        "dimensions": 4096
    },
    
    # # 本地Embedding (通过OpenAI兼容接口)
    # "local-embedding": {
    #     "type": "embedding_api",
    #     "model": "BAAI/bge-large-zh-v1.5", # 或其他本地embedding模型
    #     "api_base": "http://localhost:8001/v1",  
    #     "api_key": "EMPTY",
    #     "provider": "local",
    #     "dimensions": 1024
    # }
}

# 默认配置
DEFAULT_CONFIGS = {
    "temperature": 0.1,
    "max_tokens": 2000,
    "top_p": 0.9,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "timeout": 60.0
}

# 成本配置 (每1000 tokens的美元成本)
COST_CONFIGS = {
    "siliconflow": {
        "input_cost_per_1k": 0.0005,  # 输入token成本
        "output_cost_per_1k": 0.001   # 输出token成本
    },
    "vllm": {
        "input_cost_per_1k": 0.0,
        "output_cost_per_1k": 0.0
    },
    "openai": {
        "gpt-4": {"input_cost_per_1k": 0.03, "output_cost_per_1k": 0.06},
        "gpt-3.5-turbo": {"input_cost_per_1k": 0.001, "output_cost_per_1k": 0.002}
    }
}

# 其他配置
HUGGING_FACE_HUB_TOKEN = "XXXXXX-XXXXXX-XXXXXX-XXXXXX-XXXXXX"

def get_model_config(model_name: str) -> dict:
    """获取指定模型的配置信息"""
    if model_name not in MODEL_CONFIGS:
        available_models = list(MODEL_CONFIGS.keys())
        raise ValueError(f"模型 {model_name} 未在配置中找到。可用模型: {available_models}")
    return MODEL_CONFIGS[model_name].copy()

def get_all_llm_models() -> list:
    """获取所有可用的LLM模型列表"""
    return [name for name, config in MODEL_CONFIGS.items() if config["type"] == "llm_api"]

def get_all_embedding_models() -> list:
    """获取所有可用的Embedding模型列表"""  
    return [name for name, config in MODEL_CONFIGS.items() if config["type"] == "embedding_api"]

def get_model_cost_config(model_name: str) -> dict:
    """获取模型成本配置"""
    model_config = get_model_config(model_name)
    provider = model_config["provider"]
    
    if provider in COST_CONFIGS:
        return COST_CONFIGS[provider]
    else:
        # 返回默认成本
        return {"input_cost_per_1k": 0.001, "output_cost_per_1k": 0.002}

def get_default_config() -> dict:
    """获取默认配置"""
    return DEFAULT_CONFIGS.copy()

def validate_model_config(model_name: str) -> bool:
    """验证模型配置是否正确"""
    try:
        config = get_model_config(model_name)
        required_fields = ["type", "model", "api_base", "api_key", "provider"]
        
        for field in required_fields:
            if field not in config:
                print(f"模型配置缺少字段: {field}")
                return False
                
        if config["type"] not in ["llm_api", "embedding_api"]:
            print(f"不支持的模型类型: {config['type']}")
            return False
            
        return True
        
    except Exception as e:
        print(f"验证模型配置时出错: {e}")
        return False

def list_models_by_provider(provider: str = None) -> dict:
    """按提供商列出模型"""
    result = {}
    
    for model_name, config in MODEL_CONFIGS.items():
        model_provider = config["provider"]
        
        if provider is None or model_provider == provider:
            if model_provider not in result:
                result[model_provider] = {"llm_models": [], "embedding_models": []}
                
            if config["type"] == "llm_api":
                result[model_provider]["llm_models"].append(model_name)
            elif config["type"] == "embedding_api":
                result[model_provider]["embedding_models"].append(model_name)
    
    return result

# 便利函数：获取推荐的模型配置
def get_recommended_models() -> dict:
    """获取推荐的模型配置"""
    return {
        "default_llm": "Qwen/Qwen3-8B",
        "fast_llm": "Qwen/Qwen2.5-7B-Instruct", 
        "powerful_llm": "deepseek-ai/DeepSeek-V3",
        "default_embedding": "Qwen/Qwen3-Embedding-4B",
        "high_dim_embedding": "Qwen/Qwen3-Embedding-8B"
    }
