# FINSABER API参考文档

## 核心API

### FINSABER类

位置: `backtest/finsaber.py:19`

主要的框架入口类，用于运行回测实验。

#### 构造函数

```python
def __init__(self, trade_config: dict)
```

**参数:**
- `trade_config` (dict): 交易配置字典

**示例:**
```python
config = {
    "cash": 100000,
    "date_from": "2020-01-01",
    "date_to": "2021-01-01",
    "tickers": ["AAPL", "MSFT"],
    "data_loader": dataset_instance
}
finsaber = FINSABER(config)
```

#### 核心方法

##### run_rolling_window()

```python
def run_rolling_window(self, strategy_class, rolling_window_size=None, 
                      rolling_window_step=None, strat_params=None)
```

运行滚动窗口回测，适用于长期实验。

**参数:**
- `strategy_class`: 策略类
- `rolling_window_size` (int, optional): 滚动窗口大小（年）
- `rolling_window_step` (int, optional): 滚动步长（年）
- `strat_params` (dict): 策略参数

**返回:**
- `dict`: 评估指标字典

**示例:**
```python
from backtest.strategy.timing import BuyAndHoldStrategy

metrics = finsaber.run_rolling_window(
    BuyAndHoldStrategy,
    rolling_window_size=2,
    rolling_window_step=1,
    strat_params={}
)
```

##### run_iterative_tickers()

```python
def run_iterative_tickers(self, strategy_class, strat_params=None, 
                         tickers=None, delist_check=False)
```

对多个股票迭代运行回测。

**参数:**
- `strategy_class`: 策略类
- `strat_params` (dict, optional): 策略参数
- `tickers` (list, optional): 股票代码列表
- `delist_check` (bool): 是否检查退市

**返回:**
- `dict`: 按股票分组的评估指标

## 配置管理

### TradeConfig类

位置: `backtest/toolkit/trade_config.py`

管理所有交易配置参数。

#### 主要属性

```python
class TradeConfig:
    cash: float                    # 初始资金
    date_from: str                # 开始日期
    date_to: str                  # 结束日期 
    tickers: List[str]            # 股票代码列表
    data_loader: BacktestDataset  # 数据加载器
    risk_free_rate: float         # 无风险利率
    rolling_window_size: int      # 滚动窗口大小
    rolling_window_step: int      # 滚动步长
    save_results: bool            # 是否保存结果
    silence: bool                 # 是否静默模式
```

#### 静态方法

```python
@staticmethod
def from_dict(config_dict: dict) -> 'TradeConfig'
```

从字典创建TradeConfig实例。

## 数据接口

### BacktestDataset抽象基类

位置: `backtest/data_util/backtest_dataset.py`

所有数据集的基类接口。

#### 抽象方法

```python
@abstractmethod
def get_subset_by_time_range(self, start_date: str, end_date: str) -> 'BacktestDataset'
```

获取指定时间范围的数据子集。

```python
@abstractmethod  
def get_tickers_list(self) -> List[str]
```

获取所有可用的股票代码列表。

```python
@abstractmethod
def get_date_range(self) -> Tuple[str, str]
```

获取数据的时间范围。

### FinMemDataset实现类

位置: `backtest/data_util/finmem_dataset.py`

处理聚合数据格式的具体实现。

#### 构造函数

```python
def __init__(self, data_path: str, data_type: str = "aggregated")
```

**参数:**
- `data_path`: 数据文件路径
- `data_type`: 数据类型（"aggregated" 或 "csv"）

## 策略基类

### BaseStrategy

位置: `backtest/strategy/timing/base_strategy.py`

所有时序策略的基类。

#### 核心方法

```python
@abstractmethod
def next(self):
    """每个交易日的策略逻辑"""
    pass

def train(self):
    """训练策略模型（可选）"""
    pass
```

### BaseStrategyISO

位置: `backtest/strategy/timing_llm/base_strategy_iso.py`

LLM策略的基类，提供独立的执行环境。

#### 核心属性

```python
class BaseStrategyISO:
    symbol: str           # 股票代码
    date_from: str       # 开始日期
    date_to: str         # 结束日期
    training_years: int  # 训练年数
    data_loader: BacktestDataset  # 数据加载器
```

## 选择策略

### BaseSelector

位置: `backtest/strategy/selection/base_selector.py`

股票选择策略的基类。

#### 抽象方法

```python
@abstractmethod
def select(self, data_loader: BacktestDataset, 
           start_date: str, end_date: str) -> List[str]
```

选择股票的核心方法。

### FinMemSelector

位置: `backtest/strategy/selection/finmem_selector.py`

基于FinMem数据的选择策略实现。

## 回测框架

### FINSABERFrameworkHelper

位置: `backtest/toolkit/backtest_framework_iso.py`

底层回测执行框架。

#### 构造函数

```python
def __init__(self, initial_cash: float = 100000.0,
             risk_free_rate: float = 0.02,
             commission_per_share: float = 0.0049,
             min_commission: float = 0.99)
```

#### 主要方法

```python
def load_backtest_data_single_ticker(self, dataset: BacktestDataset,
                                   ticker: str, start_date: str, 
                                   end_date: str) -> bool
```

加载单个股票的回测数据。

```python
def run(self, strategy, delist_check: bool = False) -> bool  
```

运行回测策略。

```python
def evaluate(self, strategy) -> dict
```

评估策略性能。

**返回指标:**
- `total_return`: 总收益率
- `annual_return`: 年化收益率  
- `max_drawdown`: 最大回撤
- `annual_volatility`: 年化波动率
- `sharpe_ratio`: 夏普比率
- `sortino_ratio`: 索提诺比率
- `total_commission`: 总佣金

## 工具类

### 成本监控

```python
# 位置: backtest/toolkit/llm_cost_monitor.py

def reset_llm_cost():
    """重置LLM成本计数器"""
    
def get_llm_cost() -> float:
    """获取当前LLM调用成本"""
    
def add_llm_cost(cost: float):
    """添加LLM调用成本"""
```

### 指标计算

```python  
# 位置: backtest/toolkit/metrics.py

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float) -> float:
    """计算夏普比率"""
    
def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float) -> float:
    """计算索提诺比率"""
    
def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """计算最大回撤"""
```

### 结果聚合

```python
# 位置: backtest/toolkit/operation_utils.py

def aggregate_results_one_strategy(setup_name: str, strategy_name: str):
    """聚合单个策略的结果"""
```

## 异常处理

### 自定义异常

```python
# 位置: backtest/toolkit/custom_exceptions.py

class InsufficientTrainingDataException(Exception):
    """训练数据不足异常"""
    pass
```

## 使用示例

### 基本回测示例

```python
from backtest.finsaber import FINSABER
from backtest.data_util.finmem_dataset import FinMemDataset
from backtest.strategy.timing.buy_and_hold import BuyAndHoldStrategy

# 加载数据
dataset = FinMemDataset("path/to/data.pkl")

# 配置交易参数
config = {
    "cash": 100000,
    "date_from": "2020-01-01", 
    "date_to": "2021-01-01",
    "tickers": ["AAPL", "MSFT", "GOOGL"],
    "data_loader": dataset,
    "risk_free_rate": 0.02,
    "save_results": True,
    "silence": False
}

# 创建FINSABER实例
finsaber = FINSABER(config)

# 运行回测
results = finsaber.run_iterative_tickers(
    BuyAndHoldStrategy,
    strat_params={}
)

# 查看结果
for ticker, metrics in results.items():
    print(f"{ticker}: 年化收益率 {metrics['annual_return']:.2%}")
```

### 滚动窗口回测示例

```python
from backtest.strategy.timing.sma_crossover import SMACrossoverStrategy

# 滚动窗口回测
results = finsaber.run_rolling_window(
    SMACrossoverStrategy,
    rolling_window_size=2,
    rolling_window_step=1,
    strat_params={
        "short_window": 20,
        "long_window": 50
    }
)
```

### LLM策略示例

```python
from backtest.strategy.timing_llm.finmem import FinMemStrategy

# 加载LLM策略配置
import json
with open("strats_configs/finmem_config_normal.json", "r") as f:
    llm_config = json.load(f)

# 运行LLM策略
results = finsaber.run_iterative_tickers(
    FinMemStrategy,
    strat_params={
        "config": llm_config,
        "date_from": "$date_from",
        "date_to": "$date_to", 
        "symbol": "$symbol"
    }
)
```

## 参数解析

FINSABER支持动态参数解析，使用`$`前缀：

```python
strat_params = {
    "symbol": "$symbol",        # 自动解析为当前股票代码
    "date_from": "$date_from",  # 自动解析为开始日期
    "date_to": "$date_to"       # 自动解析为结束日期
}
```

## 结果保存格式

回测结果以pickle格式保存，结构如下：

```python
{
    "2020-01-01_2021-01-01": {
        "AAPL": {
            "total_return": 0.25,
            "annual_return": 0.24,
            "max_drawdown": -0.15,
            "sharpe_ratio": 1.2,
            "equity_with_time": pd.DataFrame
        }
    }
}
```