"""
Stable Baselines 3 模型封装，为FINSABER项目提供强化学习代理
"""

from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm


class DRLAgent:
    """深度强化学习代理的统一接口"""
    
    def __init__(self, env=None):
        self.env = env
        self.model = None
    
    def get_model(self, model_name: str, model_kwargs: dict = None, verbose: int = 1) -> BaseAlgorithm:
        """
        根据模型名称创建强化学习模型
        
        Args:
            model_name: 模型名称，支持 'a2c', 'ddpg', 'ppo', 'sac', 'td3'
            model_kwargs: 模型参数
            verbose: 日志级别
        
        Returns:
            BaseAlgorithm: 强化学习模型实例
        """
        if model_kwargs is None:
            model_kwargs = {}
            
        model_name = model_name.lower()
        
        if model_name == 'a2c':
            self.model = A2C('MlpPolicy', self.env, verbose=verbose, **model_kwargs)
        elif model_name == 'ddpg':
            self.model = DDPG('MlpPolicy', self.env, verbose=verbose, **model_kwargs)
        elif model_name == 'ppo':
            self.model = PPO('MlpPolicy', self.env, verbose=verbose, **model_kwargs)
        elif model_name == 'sac':
            self.model = SAC('MlpPolicy', self.env, verbose=verbose, **model_kwargs)
        elif model_name == 'td3':
            self.model = TD3('MlpPolicy', self.env, verbose=verbose, **model_kwargs)
        else:
            raise ValueError(f"不支持的模型类型: {model_name}")
            
        return self.model
    
    def train_model(self, total_timesteps: int = 10000, **kwargs):
        """训练模型"""
        if self.model is None:
            raise ValueError("模型未初始化，请先调用get_model方法")
        return self.model.learn(total_timesteps=total_timesteps, **kwargs)
    
    def predict(self, observation, deterministic: bool = True):
        """预测动作"""
        if self.model is None:
            raise ValueError("模型未初始化，请先调用get_model方法")
        return self.model.predict(observation, deterministic=deterministic)
    
    def save_model(self, path: str):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型未初始化，请先调用get_model方法")
        self.model.save(path)
    
    def load_model(self, path: str, model_name: str):
        """加载模型"""
        model_name = model_name.lower()
        
        if model_name == 'a2c':
            self.model = A2C.load(path, env=self.env)
        elif model_name == 'ddpg':
            self.model = DDPG.load(path, env=self.env)
        elif model_name == 'ppo':
            self.model = PPO.load(path, env=self.env)
        elif model_name == 'sac':
            self.model = SAC.load(path, env=self.env)
        elif model_name == 'td3':
            self.model = TD3.load(path, env=self.env)
        else:
            raise ValueError(f"不支持的模型类型: {model_name}")
            
        return self.model


def get_model(model_name: str, env, model_kwargs: dict = None, verbose: int = 1):
    """便利函数，快速创建模型"""
    agent = DRLAgent(env)
    return agent.get_model(model_name, model_kwargs, verbose)