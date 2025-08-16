# FINSABER

[![Arxiv link](https://img.shields.io/static/v1?label=arXiv&message=2505.07078&color=red&logo=arxiv)](https://arxiv.org/abs/2505.07078)
<a href="https://pypi.org/project/finsaber/"><img alt="PyPI" src="https://img.shields.io/pypi/v/finsaber"></a>
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

FINSABER是一个全面的交易策略评估框架，专门用于比较传统技术分析方法与现代机器学习和大语言模型(LLM)策略。该框架支持多种策略类型，提供完整的回测环境，并包含丰富的性能评估指标。

<img src="https://github.com/waylonli/FINSABER/blob/main/figs/framework.png" width="900">

## ✨ 核心特性

- 🔧 **模块化设计**: 支持技术分析、机器学习和LLM策略的统一框架
- 📊 **丰富的策略库**: 内置多种经典和前沿交易策略
- 🤖 **LLM集成**: 原生支持GPT-4等大语言模型进行智能交易
- 📈 **完整回测**: 滚动窗口、多股票、多时间段回测支持
- 🎯 **股票选择**: 多种股票选择策略，包括因子选择和AI选择
- 📋 **详细评估**: 夏普比率、最大回撤、索提诺比率等多维度指标
- 🚀 **易于扩展**: 简单的接口设计，方便开发自定义策略

## 📚 文档

**完整文档**: [docs/](docs/)

### 快速导航
- **[快速入门](docs/quickstart.md)** - 10分钟上手指南
- **[架构文档](docs/architecture.md)** - 框架设计详解  
- **[API参考](docs/api_reference.md)** - 完整API文档
- **[策略开发](docs/strategy_development.md)** - 自定义策略开发
- **[实验复现](docs/experiments.md)** - 论文实验复现指南
- **[数据格式](docs/data_format.md)** - 数据格式规范
- **[配置指南](docs/configuration.md)** - 详细配置说明

### CLI快速参考

```bash
# 基准策略回测
python backtest/run_baselines_exp.py \
    --setup selected_4 \
    --include BuyAndHoldStrategy \
    --date_from 2020-01-01 \
    --date_to 2021-01-01

# LLM策略回测  
python backtest/run_llm_traders_exp.py \
    --setup selected_4 \
    --strategy FinMemStrategy \
    --strat_config_path strats_configs/finmem_config_normal.json \
    --date_from 2020-01-01 \
    --date_to 2021-01-01

# 查看所有可用策略
ls backtest/strategy/timing/        # 传统和机器学习策略
ls backtest/strategy/timing_llm/    # LLM策略
```


## 🚀 快速开始

### 1. 安装

#### 方式一：仅安装FINSABER框架
```bash
conda create -n finsaber python=3.10
conda activate finsaber
pip install finsaber
```

#### 方式二：完整安装（推荐）
```bash
git clone https://github.com/waylonli/FINSABER
cd FINSABER
conda create -n finsaber python=3.10
conda activate finsaber
pip install -r requirements.txt
```

> ⚠️ **注意**: 如果pip安装faiss遇到问题，请使用：`conda install faiss-cpu`

### 2. 环境配置

将`.env.example`重命名为`.env`并设置环境变量：
```bash
cp .env.example .env
```

编辑`.env`文件：
```bash
OPENAI_API_KEY=your_openai_api_key_here  # LLM策略必需
HF_ACCESS_TOKEN=your_huggingface_token   # 可选
```

### 3. 第一个回测示例

```python
from backtest.finsaber import FINSABER
from backtest.data_util.finmem_dataset import FinMemDataset  
from backtest.strategy.timing.buy_and_hold import BuyAndHoldStrategy

# 加载数据
dataset = FinMemDataset("path/to/your/data.pkl")

# 配置回测参数
config = {
    "cash": 100000,
    "date_from": "2020-01-01",
    "date_to": "2021-01-01", 
    "tickers": ["AAPL", "MSFT"],
    "data_loader": dataset
}

# 运行回测
finsaber = FINSABER(config)
results = finsaber.run_iterative_tickers(BuyAndHoldStrategy)

# 查看结果
for ticker, metrics in results.items():
    print(f"{ticker}: 年化收益率 {metrics['annual_return']:.2%}")
```

## 📊 数据准备

### 预制数据集下载

我们提供了多个预处理的数据集供您直接使用：

| 数据集 | 大小 | 包含内容 | 下载链接 |
|--------|------|----------|----------|
| **S&P500聚合数据** | 10.23 GB | 价格+新闻+财报 | [下载链接](https://drive.google.com/file/d/1g9GTNr1av2b9-HphssRrQsLSnoyW0lCF/view?usp=sharing) |
| **价格数据(CSV)** | 253 MB | 仅价格数据 | [下载链接](https://drive.google.com/file/d/1KfIjn3ydynLduEYa-C5TmYud-ULkbBvM/view?usp=sharing) |
| **精选股票数据** | 48.1 MB | TSLA,AMZN,MSFT,NFLX,COIN | [下载链接](https://drive.google.com/file/d/1pmeG3NqENNW2ak_NnobG_Onu9SUSEy61/view?usp=sharing) |

### 数据格式

FINSABER支持聚合数据格式，包含多模态金融数据：

```python
{
    datetime.date(2024,1,1): {
        "price": {
            "AAPL": {"open": 150.0, "high": 155.0, "low": 149.0, 
                    "close": 154.0, "volume": 1000000},
            "MSFT": {...}
        },
        "news": {
            "AAPL": [{"title": "...", "content": "...", "sentiment": 0.8}],
            "MSFT": [...]
        },
        "filing_k": {"AAPL": {"content": "...", "summary": "..."}},
        "filing_q": {"AAPL": {"content": "...", "summary": "..."}}
    }
}
```

### 自定义数据

要使用您自己的数据，请：
1. 继承 `backtest.data_util.backtest_dataset.BacktestDataset` 类
2. 实现必要的数据访问方法
3. 参考 `backtest/data_util/finmem_dataset.py` 的实现示例

详细信息请参阅 [数据格式文档](docs/data_format.md)。


## 🔬 论文实验复现

论文包含三个主要实验设置，完整的复现指南请参阅 [实验复现文档](docs/experiments.md)。

### 实验设置概览

| 实验设置 | 描述 | 股票选择 | 测试期间 |
|----------|------|----------|----------|
| **Cherry Picking** | 选择性结果展示 | FinMem/FinCon智能体 | 2022-10-05 ~ 2023-06-10 |
| **Selected-4** | 固定4股票测试 | TSLA,AMZN,MSFT,NFLX | 2004-01-01 ~ 2024-01-01 |
| **Composite** | 多种选择策略 | 因子选择/随机选择 | 2004-01-01 ~ 2024-01-01 |

### 快速复现命令

```bash
# 1. Cherry Picking - FinMem设置
python backtest/run_llm_traders_exp.py \
    --setup cherry_pick_both_finmem \
    --strategy FinMemStrategy \
    --strat_config_path strats_configs/finmem_config_cherry.json \
    --date_from 2022-10-05 --date_to 2023-06-10

# 2. Selected-4 - 买入持有基准
python backtest/run_baselines_exp.py \
    --setup selected_4 \
    --include BuyAndHoldStrategy \
    --date_from 2004-01-01 --date_to 2024-01-01 \
    --rolling_window_size 2

# 3. Composite - 动量因子选择
python backtest/run_baselines_exp.py \
    --setup momentum_sp500_5 \
    --include SMACrossStrategy \
    --date_from 2004-01-01 --date_to 2024-01-01 \
    --training_years 2
```

### 支持的策略类型

#### 传统策略 (`backtest/strategy/timing/`)
- `BuyAndHoldStrategy` - 买入持有
- `SMACrossStrategy` - 简单移动平均交叉
- `BollingerBandsStrategy` - 布林带策略
- `XGBoostPredictorStrategy` - XGBoost预测
- 等等...

#### LLM策略 (`backtest/strategy/timing_llm/`)
- `FinMemStrategy` - 基于记忆增强的LLM策略
- `FinAgentStrategy` - 基于智能体的LLM策略

完整的策略列表和参数设置请参考 [策略开发文档](docs/strategy_development.md)。

## 🏗️ 项目结构

```
FINSABER/
├── docs/                     # 📚 完整文档
│   ├── quickstart.md        # 快速入门指南  
│   ├── architecture.md      # 架构设计文档
│   ├── api_reference.md     # API参考手册
│   ├── strategy_development.md # 策略开发指南
│   └── experiments.md       # 实验复现指南
├── backtest/                # 🔧 核心回测框架
│   ├── strategy/            # 策略实现
│   │   ├── timing/         # 传统和机器学习策略
│   │   └── timing_llm/     # LLM策略
│   ├── data_util/          # 数据处理工具
│   └── toolkit/            # 回测工具包
├── llm_traders/            # 🤖 LLM交易员实现
│   ├── finagent/           # FinAgent策略
│   └── finmem/             # FinMem策略
└── strats_configs/         # ⚙️ 策略配置文件
```

## 🤝 贡献

我们欢迎各种形式的贡献！请参考以下方式：

1. **🐛 Bug报告**: 在 [GitHub Issues](https://github.com/waylonli/FINSABER/issues) 中报告问题
2. **💡 功能建议**: 提出新功能或改进建议
3. **📝 文档改进**: 帮助完善文档内容
4. **🔧 代码贡献**: 提交Pull Request来改进代码

### 开发指南
1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 📖 引用

如果您在研究中使用了FINSABER，请引用我们的论文：

```bibtex
@misc{li2025llmbasedfinancialinvestingstrategies,
    title={Can LLM-based Financial Investing Strategies Outperform the Market in Long Run?}, 
    author={Weixian Waylon Li and Hyeonjun Kim and Mihai Cucuringu and Tiejun Ma},
    year={2025},
    eprint={2505.07078},
    archivePrefix={arXiv},
    primaryClass={q-fin.TR},
    url={https://arxiv.org/abs/2505.07078}, 
}
```

## 📞 联系我们

- **GitHub Issues**: [问题反馈](https://github.com/waylonli/FINSABER/issues)
- **论文链接**: [arXiv:2505.07078](https://arxiv.org/abs/2505.07078)
- **PyPI包**: [finsaber](https://pypi.org/project/finsaber/)

---

⭐ 如果这个项目对您有帮助，请给我们一个星标！
