# 配置指南

本文档详细说明FINSABER框架的各种配置选项和参数设置。

## 环境配置

### 环境变量

在`.env`文件中设置必要的环境变量：

```bash
# LLM API配置
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_ORG_ID=org-your-org-id-here              # 可选
OPENAI_BASE_URL=https://api.openai.com/v1       # 可选，使用代理时修改

# Hugging Face配置  
HF_ACCESS_TOKEN=hf_your-hugging-face-token      # 可选

# 其他API配置
ALPHA_VANTAGE_API_KEY=your-alpha-vantage-key    # 可选
FINNHUB_API_KEY=your-finnhub-key                # 可选
POLYGON_API_KEY=your-polygon-key                # 可选

# 数据路径配置
DATA_ROOT_PATH=/path/to/your/data               # 数据根目录
OUTPUT_ROOT_PATH=/path/to/output                # 输出根目录

# 日志配置
LOG_LEVEL=INFO                                  # DEBUG, INFO, WARNING, ERROR
LOG_FILE_PATH=/path/to/logs/finsaber.log       # 日志文件路径

# 代理配置（如果需要）
HTTP_PROXY=http://proxy.company.com:8080
HTTPS_PROXY=http://proxy.company.com:8080
```

### Python环境配置

```python
# config.py - 全局配置文件
import os
from typing import Dict, Any

# 基础配置
BASE_CONFIG = {
    "data_root": os.getenv("DATA_ROOT_PATH", "./data"),
    "output_root": os.getenv("OUTPUT_ROOT_PATH", "./backtest/output"),
    "log_level": os.getenv("LOG_LEVEL", "INFO"),
    "log_file": os.getenv("LOG_FILE_PATH", "finsaber.log")
}

# LLM配置
LLM_CONFIG = {
    "openai": {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "org_id": os.getenv("OPENAI_ORG_ID"),
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "default_model": "gpt-4",
        "timeout": 60,
        "max_retries": 3
    }
}

# 数据源配置
DATA_SOURCE_CONFIG = {
    "default_source": "local",
    "sources": {
        "alpha_vantage": {
            "api_key": os.getenv("ALPHA_VANTAGE_API_KEY"),
            "base_url": "https://www.alphavantage.co/query"
        },
        "finnhub": {
            "api_key": os.getenv("FINNHUB_API_KEY"),
            "base_url": "https://finnhub.io/api/v1"
        }
    }
}
```

## 交易配置

### TradeConfig类配置

```python
from backtest.toolkit.trade_config import TradeConfig

# 基础交易配置
basic_config = {
    # 资金配置
    "cash": 100000,                    # 初始资金
    "commission": 0.0049,              # 每股佣金
    "min_commission": 0.99,            # 最小佣金
    "risk_free_rate": 0.02,            # 无风险利率
    
    # 时间配置
    "date_from": "2020-01-01",         # 开始日期
    "date_to": "2021-01-01",           # 结束日期
    
    # 股票配置
    "tickers": ["AAPL", "MSFT", "GOOGL"],  # 股票列表
    "setup_name": "selected_stocks",    # 实验设置名称
    
    # 滚动窗口配置
    "rolling_window_size": 2,          # 滚动窗口大小（年）
    "rolling_window_step": 1,          # 滚动步长（年）
    
    # 输出配置
    "save_results": True,              # 是否保存结果
    "result_filename": None,           # 自定义结果文件名
    "log_base_dir": "backtest/output", # 日志基础目录
    "silence": False                   # 是否静默模式
}

# 创建配置实例
trade_config = TradeConfig.from_dict(basic_config)
```

### 高级交易配置

```python
# 高级配置选项
advanced_config = {
    # 基础配置
    **basic_config,
    
    # 风险管理
    "max_position_size": 0.2,          # 最大单个持仓比例
    "max_total_leverage": 1.0,         # 最大总杠杆
    "stop_loss_pct": 0.05,             # 止损百分比
    "take_profit_pct": 0.15,           # 止盈百分比
    
    # 数据配置
    "data_loader": None,               # 数据加载器实例
    "validate_data": True,             # 是否验证数据
    "min_data_points": 252,            # 最少数据点数（约1年）
    
    # 选择策略配置
    "selection_strategy": None,        # 股票选择策略实例
    "rebalance_frequency": "monthly",  # 重新平衡频率
    
    # 性能配置
    "parallel_execution": True,        # 是否并行执行
    "max_workers": 4,                  # 最大工作进程数
    "chunk_size": 10,                  # 批处理大小
    
    # 缓存配置
    "enable_cache": True,              # 是否启用缓存
    "cache_dir": "./cache",            # 缓存目录
    "cache_expiry": 86400              # 缓存过期时间（秒）
}
```

## 策略配置

### 技术分析策略配置

```python
# SMA交叉策略配置
sma_config = {
    "strategy_class": "SMACrossStrategy",
    "parameters": {
        "short_window": 20,            # 短期窗口
        "long_window": 50,             # 长期窗口
        "use_ema": False               # 是否使用指数移动平均
    }
}

# 布林带策略配置
bollinger_config = {
    "strategy_class": "BollingerBandsStrategy", 
    "parameters": {
        "window": 20,                  # 移动平均窗口
        "num_std": 2.0,                # 标准差倍数
        "use_percentage": True         # 是否使用百分比带宽
    }
}

# RSI策略配置
rsi_config = {
    "strategy_class": "RSIStrategy",
    "parameters": {
        "rsi_period": 14,              # RSI计算周期
        "oversold_level": 30,          # 超卖水平
        "overbought_level": 70,        # 超买水平
        "smoothing_period": 3          # 平滑周期
    }
}
```

### 机器学习策略配置

```python
# XGBoost预测策略配置
xgb_config = {
    "strategy_class": "XGBoostPredictorStrategy",
    "parameters": {
        "training_years": 3,           # 训练年数
        "feature_lookback": 30,        # 特征回看窗口
        "prediction_horizon": 5,       # 预测时间跨度
        "model_params": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42
        },
        "retrain_frequency": 252,      # 重新训练频率（天）
        "feature_selection": True,     # 是否进行特征选择
        "cross_validation": True       # 是否使用交叉验证
    }
}

# 强化学习策略配置
rl_config = {
    "strategy_class": "FinRLStrategy",
    "parameters": {
        "algorithm": "PPO",            # RL算法: PPO, A2C, SAC, TD3, DDPG
        "training_timesteps": 50000,   # 训练时步数
        "learning_rate": 3e-4,         # 学习率
        "environment_config": {
            "lookback_window": 30,
            "action_space": "discrete",  # discrete 或 continuous
            "reward_function": "sharpe", # sharpe, return, profit
            "transaction_cost": 0.001
        },
        "model_save_path": "./models/rl_models"
    }
}
```

### LLM策略配置

#### FinMem策略配置

```json
{
    "model_config": {
        "model": "gpt-4",
        "temperature": 0.1,
        "max_tokens": 2000,
        "top_p": 0.9,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0
    },
    "memory_config": {
        "enable_memory": true,
        "memory_type": "faiss",
        "memory_window": 30,
        "memory_decay": 0.99,
        "max_memory_items": 1000,
        "embedding_model": "text-embedding-ada-002"
    },
    "reflection_config": {
        "enable_reflection": true,
        "reflection_threshold": 0.8,
        "reflection_frequency": "daily",
        "max_reflections": 10
    },
    "trading_config": {
        "decision_confidence_threshold": 0.7,
        "max_position_size": 1.0,
        "rebalance_threshold": 0.1,
        "risk_tolerance": "moderate"
    },
    "data_config": {
        "use_news": true,
        "use_financials": true,
        "use_technical": true,
        "news_lookback_days": 7,
        "max_news_items": 20
    }
}
```

#### FinAgent策略配置

```json
{
    "agent_config": {
        "model": "gpt-4",
        "temperature": 0.1,
        "system_prompt_template": "trading_expert",
        "enable_tools": true,
        "tool_timeout": 30
    },
    "tools_config": {
        "technical_analysis": {
            "enabled": true,
            "indicators": ["sma", "rsi", "bollinger", "macd"]
        },
        "fundamental_analysis": {
            "enabled": true,
            "metrics": ["pe_ratio", "book_value", "debt_ratio"]
        },
        "sentiment_analysis": {
            "enabled": true,
            "sources": ["news", "social_media"],
            "aggregation_method": "weighted_average"
        },
        "market_data": {
            "enabled": true,
            "real_time": false,
            "data_sources": ["yahoo", "alpha_vantage"]
        }
    },
    "decision_config": {
        "multi_step_reasoning": true,
        "confidence_scoring": true,
        "explanation_required": true,
        "decision_tree_depth": 3
    },
    "risk_management": {
        "max_drawdown": 0.15,
        "position_sizing": "kelly",
        "stop_loss_strategy": "trailing",
        "portfolio_diversification": true
    }
}
```

## 数据配置

### 数据源配置

```python
# 数据加载器配置
data_config = {
    "source_type": "aggregated",       # aggregated 或 csv
    "data_path": "./data/sp500_data.pkl",
    "cache_enabled": True,
    "cache_dir": "./data/cache",
    
    # 数据验证配置
    "validation": {
        "check_missing_dates": True,
        "check_price_continuity": True,
        "check_volume_sanity": True,
        "outlier_detection": True
    },
    
    # 数据预处理配置
    "preprocessing": {
        "fill_missing": "forward_fill",  # forward_fill, backward_fill, interpolate
        "outlier_handling": "winsorize",  # remove, winsorize, cap
        "normalization": None,            # z_score, min_max, robust
        "feature_engineering": True
    }
}

# 创建数据加载器
from backtest.data_util.finmem_dataset import FinMemDataset
dataset = FinMemDataset(
    data_path=data_config["data_path"],
    data_type=data_config["source_type"]
)
```

### 股票选择器配置

```python
# 动量因子选择器配置
momentum_selector_config = {
    "selector_class": "MomentumFactorSP500Selector",
    "parameters": {
        "lookback_period": 252,        # 动量计算期间
        "selection_count": 5,          # 选择股票数量
        "rebalance_frequency": "monthly", # 重新选择频率
        "momentum_metric": "return",    # return, risk_adjusted_return
        "exclude_sectors": [],          # 排除的行业
        "min_market_cap": 1e9          # 最小市值要求
    }
}

# 低波动率选择器配置
low_vol_selector_config = {
    "selector_class": "LowVolatilitySP500Selector",
    "parameters": {
        "volatility_window": 252,      # 波动率计算窗口
        "selection_count": 5,          # 选择股票数量
        "volatility_metric": "std",    # std, var, range
        "risk_adjustment": True,       # 是否风险调整
        "correlation_threshold": 0.7   # 相关性阈值
    }
}

# FinMem智能选择器配置
finmem_selector_config = {
    "selector_class": "FinMemSelector",
    "parameters": {
        "model": "gpt-4",
        "selection_criteria": [
            "fundamental_analysis",
            "technical_momentum", 
            "market_sentiment"
        ],
        "selection_count": 5,
        "confidence_threshold": 0.8,
        "use_memory": True,
        "memory_window": 90
    }
}
```

## 系统配置

### 日志配置

```python
import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s"
        }
    },
    "handlers": {
        "console": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "level": "DEBUG", 
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "detailed",
            "filename": "finsaber.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        }
    },
    "loggers": {
        "": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": False
        },
        "backtest": {
            "handlers": ["console", "file"], 
            "level": "INFO",
            "propagate": False
        },
        "llm_traders": {
            "handlers": ["file"],
            "level": "DEBUG",
            "propagate": False
        }
    }
}

# 应用日志配置
logging.config.dictConfig(LOGGING_CONFIG)
```

### 性能配置

```python
# 性能优化配置
PERFORMANCE_CONFIG = {
    # 并行处理配置
    "parallel_processing": {
        "enabled": True,
        "max_workers": None,           # None表示使用系统最优值
        "chunk_size": 10,              # 任务分块大小
        "backend": "multiprocessing"   # multiprocessing, threading
    },
    
    # 内存管理配置
    "memory_management": {
        "max_memory_usage": "8GB",     # 最大内存使用量
        "garbage_collection": True,    # 是否启用垃圾回收
        "memory_profiling": False      # 是否启用内存分析
    },
    
    # 缓存配置
    "caching": {
        "enabled": True,
        "cache_type": "disk",          # memory, disk, redis
        "ttl": 3600,                   # 缓存过期时间（秒）
        "max_size": "1GB"              # 最大缓存大小
    },
    
    # 数据库配置（可选）
    "database": {
        "enabled": False,
        "type": "sqlite",              # sqlite, postgresql, mysql
        "connection_string": "sqlite:///finsaber.db",
        "pool_size": 10,
        "max_overflow": 20
    }
}
```

## 配置文件管理

### 配置文件结构

```
config/
├── base.json                 # 基础配置
├── development.json          # 开发环境配置
├── production.json           # 生产环境配置
├── strategies/
│   ├── technical.json        # 技术分析策略配置
│   ├── ml.json              # 机器学习策略配置
│   └── llm.json             # LLM策略配置
└── experiments/
    ├── cherry_picking.json   # 樱桃选择实验配置
    ├── selected_4.json       # 选择4股票实验配置
    └── composite.json        # 组合实验配置
```

### 配置管理类

```python
import json
import os
from typing import Dict, Any
from pathlib import Path

class ConfigManager:
    def __init__(self, config_dir: str = "./config"):
        self.config_dir = Path(config_dir)
        self.configs = {}
        
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """加载配置文件"""
        config_path = self.config_dir / f"{config_name}.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        # 处理配置继承
        if "inherit" in config:
            base_config = self.load_config(config["inherit"])
            config = self._merge_configs(base_config, config)
            del config["inherit"]
            
        self.configs[config_name] = config
        return config
        
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """合并配置"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
                
        return result
        
    def save_config(self, config_name: str, config: Dict[str, Any]):
        """保存配置文件"""
        config_path = self.config_dir / f"{config_name}.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            
        self.configs[config_name] = config
        
    def get_config(self, config_name: str) -> Dict[str, Any]:
        """获取配置（缓存优先）"""
        if config_name not in self.configs:
            return self.load_config(config_name)
        return self.configs[config_name]
        
    def validate_config(self, config_name: str, schema: Dict[str, Any]) -> bool:
        """验证配置格式"""
        config = self.get_config(config_name)
        # 实现配置验证逻辑
        return True

# 全局配置管理器
config_manager = ConfigManager()
```

### 使用配置

```python
# 加载并使用配置
from config_manager import config_manager

# 加载交易配置
trade_config = config_manager.get_config("base")
trade_config.update(config_manager.get_config("experiments/selected_4"))

# 加载策略配置
strategy_config = config_manager.get_config("strategies/llm")

# 创建FINSABER实例
finsaber = FINSABER(trade_config)

# 运行策略
results = finsaber.run_iterative_tickers(
    FinMemStrategy,
    strat_params=strategy_config["finmem"]
)
```

## 配置最佳实践

### 1. 环境分离

为不同环境使用不同的配置文件：

```python
import os

# 根据环境加载配置
env = os.getenv("ENVIRONMENT", "development")
config = config_manager.get_config(env)
```

### 2. 敏感信息管理

永远不要在配置文件中硬编码敏感信息：

```json
{
    "api_key": "${OPENAI_API_KEY}",
    "database_url": "${DATABASE_URL}"
}
```

### 3. 配置验证

在运行前验证配置的完整性：

```python
def validate_trading_config(config):
    required_fields = ["cash", "date_from", "date_to", "data_loader"]
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"配置中缺少必要字段: {field}")
            
    if config["cash"] <= 0:
        raise ValueError("初始资金必须大于0")
        
    # 更多验证逻辑...
```

### 4. 配置文档化

为每个配置选项提供清晰的文档：

```json
{
    "_comments": {
        "cash": "初始资金，单位：美元",
        "commission": "每股佣金费用",
        "risk_free_rate": "无风险利率，用于计算夏普比率"
    },
    "cash": 100000,
    "commission": 0.0049,
    "risk_free_rate": 0.02
}
```

通过合理的配置管理，可以让FINSABER框架更加灵活和易于使用。