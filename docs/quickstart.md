# FINSABER 快速入门指南

## 安装与配置

### 1. 环境安装

#### 方式一：仅安装FINSABER框架

```bash
conda create -n finsaber python=3.10
conda activate finsaber  
pip install finsaber
```

#### 方式二：完整安装（包含所有实验依赖）

```bash
git clone https://github.com/waylonli/FINSABER
cd FINSABER
conda create -n finsaber python=3.10
conda activate finsaber
pip install -r requirements.txt
```

**注意**: 如果pip安装faiss包遇到问题，请使用conda安装：
```bash
conda install faiss-cpu
```

### 2. 环境变量配置

复制 `.env.example` 为 `.env` 并配置：

```bash
cp .env.example .env
```

编辑 `.env` 文件：
```bash
OPENAI_API_KEY=your_openai_api_key_here  # 运行LLM策略必需
HF_ACCESS_TOKEN=your_huggingface_token   # 可选
```

## 数据准备

### 数据格式

FINSABER支持两种数据格式：

#### 1. 聚合数据格式（推荐）
```python
{
    datetime.date(2024,1,1): {
        "price": {"AAPL": price_data, "MSFT": price_data},
        "news": {"AAPL": news_data, "MSFT": news_data}, 
        "filing_k": {"AAPL": filing_data, "MSFT": filing_data},
        "filing_q": {"AAPL": filing_data, "MSFT": filing_data}
    }
}
```

#### 2. CSV格式（仅价格数据）
标准的OHLCV格式CSV文件。

### 数据下载

- [聚合S&P500数据](https://drive.google.com/file/d/1g9GTNr1av2b9-HphssRrQsLSnoyW0lCF/view?usp=sharing) (10.23 GB)
- [CSV格式价格数据](https://drive.google.com/file/d/1KfIjn3ydynLduEYa-C5TmYud-ULkbBvM/view?usp=sharing) (253 MB)  
- [精选股票数据](https://drive.google.com/file/d/1pmeG3NqENNW2ak_NnobG_Onu9SUSEy61/view?usp=sharing) (48.1 MB)

## 第一个回测实验

### 1. 基础回测示例

```python
from backtest.finsaber import FINSABER
from backtest.data_util.finmem_dataset import FinMemDataset
from backtest.strategy.timing.buy_and_hold import BuyAndHoldStrategy

# 加载数据
dataset = FinMemDataset("path/to/your/data.pkl")

# 配置交易参数
trade_config = {
    "cash": 100000,                    # 初始资金
    "date_from": "2020-01-01",         # 开始日期
    "date_to": "2021-01-01",           # 结束日期
    "tickers": ["AAPL", "MSFT"],       # 股票列表
    "data_loader": dataset,             # 数据加载器
    "risk_free_rate": 0.02,            # 无风险利率
    "save_results": True,              # 保存结果
    "silence": False                   # 显示详细输出
}

# 创建FINSABER实例并运行回测
finsaber = FINSABER(trade_config)
results = finsaber.run_iterative_tickers(BuyAndHoldStrategy)

# 查看结果
print("回测结果:")
for ticker, metrics in results.items():
    print(f"{ticker}:")
    print(f"  总收益率: {metrics['total_return']:.2%}")
    print(f"  年化收益率: {metrics['annual_return']:.2%}")
    print(f"  最大回撤: {metrics['max_drawdown']:.2%}")
    print(f"  夏普比率: {metrics['sharpe_ratio']:.3f}")
```

### 2. 技术分析策略示例

```python  
from backtest.strategy.timing.sma_crossover import SMACrossoverStrategy

# SMA交叉策略参数
sma_params = {
    "short_window": 20,   # 短期移动平均窗口
    "long_window": 50     # 长期移动平均窗口
}

# 运行SMA策略
results = finsaber.run_iterative_tickers(
    SMACrossoverStrategy, 
    strat_params=sma_params
)
```

### 3. 机器学习策略示例

```python
from backtest.strategy.timing.xgboost_predictor import XGBoostPredictorStrategy

# XGBoost策略参数
ml_params = {
    "training_years": 2,           # 训练数据年数
    "feature_lookback": 30,        # 特征回看窗口
    "prediction_threshold": 0.6    # 预测阈值
}

# 运行机器学习策略
results = finsaber.run_iterative_tickers(
    XGBoostPredictorStrategy,
    strat_params=ml_params
)
```

## 运行LLM策略

### 1. 准备LLM策略配置

创建配置文件 `my_finmem_config.json`：

```json
{
    "model": "gpt-4",
    "temperature": 0.1,
    "max_tokens": 1000,
    "memory_settings": {
        "enable_memory": true,
        "memory_window": 30
    },
    "trading_settings": {
        "max_position_size": 0.1,
        "stop_loss": 0.05
    }
}
```

### 2. 运行FinMem策略

```python
from backtest.strategy.timing_llm.finmem import FinMemStrategy
import json

# 加载LLM配置
with open("my_finmem_config.json", "r") as f:
    llm_config = json.load(f)

# LLM策略参数（使用动态参数解析）
llm_params = {
    "config": llm_config,
    "date_from": "$date_from",  # 自动解析为配置中的开始日期
    "date_to": "$date_to",      # 自动解析为配置中的结束日期  
    "symbol": "$symbol",        # 自动解析为当前股票代码
    "training_years": 2
}

# 运行LLM策略
results = finsaber.run_iterative_tickers(
    FinMemStrategy,
    strat_params=llm_params
)
```

## 滚动窗口回测

对于长期实验，可以使用滚动窗口回测：

```python
# 更新配置为长期数据
trade_config.update({
    "date_from": "2004-01-01", 
    "date_to": "2024-01-01"
})

finsaber = FINSABER(trade_config)

# 滚动窗口回测
rolling_results = finsaber.run_rolling_window(
    BuyAndHoldStrategy,
    rolling_window_size=2,    # 2年窗口
    rolling_window_step=1,    # 1年步长  
    strat_params={}
)
```

## 股票选择策略

### 使用预定义选择策略

```python
from backtest.strategy.selection.momentum_factor_sp500_selector import MomentumFactorSP500Selector

# 在配置中指定选择策略
trade_config["selection_strategy"] = MomentumFactorSP500Selector()
trade_config["setup_name"] = "momentum_sp500_5"  # 对应选择策略名称
```

### 自定义选择策略

```python
from backtest.strategy.selection.base_selector import BaseSelector

class MyCustomSelector(BaseSelector):
    def select(self, data_loader, start_date, end_date):
        # 实现您的选择逻辑
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
```

## 命令行运行

### 基础策略实验

```bash
python backtest/run_baselines_exp.py \
    --setup selected_4 \
    --include BuyAndHoldStrategy \
    --date_from 2020-01-01 \
    --date_to 2021-01-01 \
    --training_years 2 \
    --rolling_window_size 1 \
    --rolling_window_step 1
```

### LLM策略实验  

```bash
python backtest/run_llm_traders_exp.py \
    --setup cherry_pick_both_finmem \
    --strategy FinMemStrategy \
    --strat_config_path strats_configs/finmem_config_normal.json \
    --date_from 2020-01-01 \
    --date_to 2021-01-01 \
    --rolling_window_size 1 \
    --rolling_window_step 1
```

## 结果分析

### 查看保存的结果

```python
import pickle
import pandas as pd

# 加载结果
with open("backtest/output/selected_4/BuyAndHoldStrategy/2020-01-01_2021-01-01.pkl", "rb") as f:
    results = pickle.load(f)

# 创建结果汇总表
summary_data = []
for period, tickers_data in results.items():
    for ticker, metrics in tickers_data.items():
        summary_data.append({
            "Period": period,
            "Ticker": ticker,  
            "Annual Return": f"{metrics['annual_return']:.2%}",
            "Max Drawdown": f"{metrics['max_drawdown']:.2%}",
            "Sharpe Ratio": f"{metrics['sharpe_ratio']:.3f}"
        })

df = pd.DataFrame(summary_data)
print(df)
```

### 绘制权益曲线

```python
import matplotlib.pyplot as plt

# 获取特定股票的权益曲线
ticker = "AAPL"
period = "2020-01-01_2021-01-01" 
equity_data = results[period][ticker]["equity_with_time"]

# 绘制权益曲线
plt.figure(figsize=(12, 6))
plt.plot(equity_data["datetime"], equity_data["equity"])
plt.title(f"{ticker} 权益曲线")
plt.xlabel("日期")
plt.ylabel("权益价值")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
```

## 常见问题解决

### 1. 数据不足错误
```python
# 捕获并处理数据不足异常
from backtest.toolkit.custom_exceptions import InsufficientTrainingDataException

try:
    results = finsaber.run_iterative_tickers(strategy_class)
except InsufficientTrainingDataException as e:
    print(f"训练数据不足: {e}")
    # 调整训练参数或数据范围
```

### 2. LLM调用成本监控
```python
from backtest.toolkit.llm_cost_monitor import get_llm_cost, reset_llm_cost

# 重置成本计数器
reset_llm_cost()

# 运行策略
results = finsaber.run_iterative_tickers(FinMemStrategy, strat_params=llm_params)

# 查看总成本
print(f"LLM调用总成本: ${get_llm_cost():.2f}")
```

### 3. 内存优化
```python
# 对于大型数据集，分批处理股票
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
batch_size = 2

for i in range(0, len(tickers), batch_size):
    batch_tickers = tickers[i:i+batch_size]
    batch_config = trade_config.copy()
    batch_config["tickers"] = batch_tickers
    
    batch_finsaber = FINSABER(batch_config)
    batch_results = batch_finsaber.run_iterative_tickers(strategy_class)
```

## 下一步

- 阅读[架构文档](architecture.md)了解框架设计
- 查看[API参考](api_reference.md)了解详细接口
- 浏览[策略开发指南](strategy_development.md)学习自定义策略
- 参考[实验复现指南](experiments.md)重现论文结果