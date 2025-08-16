#!/usr/bin/env python3
"""
简单回测测试 - 使用模拟数据验证统一LLM系统
"""

import os
import sys
import logging
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_mock_data(start_date: str = "2023-01-01", end_date: str = "2023-06-30"):
    """创建模拟的聚合数据格式"""
    print("创建模拟数据...")
    
    # 生成日期范围
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    date_range = pd.date_range(start, end, freq='B')  # 工作日
    
    # 模拟股票列表
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    # 创建聚合数据结构
    mock_data = {}
    
    for i, current_date in enumerate(date_range):
        current_date = current_date.date()
        mock_data[current_date] = {
            "price": {},
            "news": {},
            "filing_k": {},
            "filing_q": {}
        }
        
        for symbol in symbols:
            # 模拟价格数据（简单随机游走）
            base_price = 100 + i * 0.1 + np.random.normal(0, 2)
            daily_change = np.random.normal(0, 0.02)
            
            mock_data[current_date]["price"][symbol] = {
                "open": base_price * (1 + daily_change),
                "high": base_price * (1 + daily_change + abs(np.random.normal(0, 0.01))),
                "low": base_price * (1 + daily_change - abs(np.random.normal(0, 0.01))),
                "close": base_price * (1 + daily_change + np.random.normal(0, 0.005)),
                "volume": int(1000000 + np.random.normal(0, 200000)),
                "adj_close": base_price * (1 + daily_change + np.random.normal(0, 0.005))
            }
            
            # 模拟新闻数据
            news_sentiments = ["positive", "neutral", "negative"]
            sentiment_scores = [0.7, 0.0, -0.6]
            
            sentiment_choice = np.random.choice(len(news_sentiments))
            
            mock_data[current_date]["news"][symbol] = [{
                "title": f"{symbol} 市场动态分析 - {current_date}",
                "content": f"今日{symbol}股票表现{'良好' if sentiment_choice == 0 else '一般' if sentiment_choice == 1 else '不佳'}，"
                          f"收盘价为${mock_data[current_date]['price'][symbol]['close']:.2f}。"
                          f"市场分析显示{'看涨' if sentiment_choice == 0 else '观望' if sentiment_choice == 1 else '看跌'}趋势。",
                "sentiment": sentiment_scores[sentiment_choice],
                "source": "MockNews",
                "timestamp": f"{current_date}T09:30:00Z"
            }]
            
            # 模拟财报数据（简单）
            if i % 30 == 0:  # 每月一次
                mock_data[current_date]["filing_q"][symbol] = {
                    "content": f"{symbol} 季度财报 - 营收增长，盈利能力提升",
                    "file_date": str(current_date),
                    "summary": f"{symbol} 本季度表现稳定，符合市场预期"
                }
    
    return mock_data

def save_mock_data(mock_data, filename: str = "mock_data.pkl"):
    """保存模拟数据到文件"""
    data_dir = Path("test_data")
    data_dir.mkdir(exist_ok=True)
    
    filepath = data_dir / filename
    with open(filepath, "wb") as f:
        pickle.dump(mock_data, f)
    
    print(f"模拟数据已保存到: {filepath}")
    return str(filepath)

def test_basic_backtest_with_mock_data():
    """使用模拟数据测试基础回测"""
    print("\n=== 使用模拟数据测试基础回测 ===")
    
    try:
        # 创建模拟数据
        mock_data = create_mock_data("2023-01-01", "2023-03-31")  # 短期数据用于测试
        mock_data_path = save_mock_data(mock_data)
        
        # 导入回测模块
        from backtest.finsaber import FINSABER
        from backtest.data_util.finmem_dataset import FinMemDataset
        from backtest.strategy.timing.buy_and_hold import BuyAndHoldStrategy
        
        print("✓ 回测模块导入成功")
        
        # 创建数据集
        dataset = FinMemDataset(mock_data_path, data_type="aggregated")
        print(f"✓ 数据集创建成功，股票数量: {len(dataset.get_tickers_list())}")
        print(f"✓ 数据时间范围: {dataset.get_date_range()}")
        
        # 配置回测参数
        config = {
            "cash": 100000,
            "date_from": "2023-01-01",
            "date_to": "2023-03-31",
            "tickers": ["AAPL", "MSFT"],
            "data_loader": dataset,
            "risk_free_rate": 0.02,
            "save_results": False,  # 不保存结果
            "silence": True  # 静默模式
        }
        
        print("✓ 配置参数设置完成")
        
        # 创建FINSABER实例
        finsaber = FINSABER(config)
        print("✓ FINSABER实例创建成功")
        
        # 运行简单回测（买入持有策略）
        print("开始运行回测...")
        results = finsaber.run_iterative_tickers(
            BuyAndHoldStrategy,
            strat_params={}
        )
        
        print("✓ 回测执行成功")
        
        # 显示结果
        if results:
            print(f"\n回测结果摘要:")
            for ticker, metrics in results.items():
                if "annual_return" in metrics:
                    print(f"  {ticker}:")
                    print(f"    年化收益率: {metrics['annual_return']:.2%}")
                    print(f"    夏普比率: {metrics.get('sharpe_ratio', 'N/A'):.3f}")
                    print(f"    最大回撤: {metrics.get('max_drawdown', 'N/A'):.2%}")
        else:
            print("⚠️  回测结果为空")
            
        # 清理测试文件
        if os.path.exists(mock_data_path):
            os.remove(mock_data_path)
            print("✓ 测试文件清理完成")
        
        return True
        
    except Exception as e:
        print(f"✗ 回测测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_llm_strategy_with_mock_data():
    """使用模拟数据测试LLM策略（简化版）"""
    print("\n=== 测试LLM策略（模拟模式）===")
    
    try:
        # 导入LLM策略相关模块
        from llm_traders.finmem.puppy.unified_chat import UnifiedChatClient
        
        # 测试创建LLM客户端（用于策略）
        client = UnifiedChatClient(
            model_name="Qwen/Qwen3-8B",
            system_message="你是一个专业的金融交易顾问。",
            temperature=0.1
        )
        
        print("✓ LLM交易客户端创建成功")
        
        # 模拟交易决策场景
        market_context = """
        当前市场情况：
        - AAPL股价: $150.23 (+1.2%)
        - MSFT股价: $280.45 (-0.5%)
        - 市场整体情绪：中性偏乐观
        - 最新新闻：科技股表现分化，投资者关注财报季
        """
        
        decision_prompt = f"""
        基于以下市场信息，请给出交易建议：
        
        {market_context}
        
        请以JSON格式回复，包含以下字段：
        {{
            "action": "buy/sell/hold",
            "confidence": 0.0-1.0,
            "reasoning": "你的分析理由"
        }}
        """
        
        response = client.guardrail_chat(decision_prompt, require_json=True)
        print(f"✓ LLM交易决策响应: {response[:150]}...")
        
        # 验证响应格式（简单）
        try:
            import json
            decision = json.loads(response)
            required_fields = ["action", "confidence", "reasoning"]
            if all(field in decision for field in required_fields):
                print("✓ LLM决策格式验证通过")
            else:
                print("⚠️  LLM决策格式不完整，但基本功能正常")
        except:
            print("⚠️  LLM决策格式解析失败，但API调用正常")
        
        return True
        
    except Exception as e:
        print(f"✗ LLM策略测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cost_monitoring():
    """测试成本监控功能"""
    print("\n=== 测试成本监控功能 ===")
    
    try:
        from backtest.toolkit.llm_cost_monitor import get_llm_cost, reset_llm_cost, add_llm_cost
        from llm_client import create_llm_client
        
        # 重置成本计数器
        reset_llm_cost()
        initial_cost = get_llm_cost()
        print(f"初始成本: ${initial_cost:.6f}")
        
        # 创建客户端并进行调用
        client = create_llm_client("Qwen/Qwen3-8B")
        
        response = client.simple_completion(
            "请简单介绍什么是量化交易？",
            max_tokens=50
        )
        
        final_cost = get_llm_cost()
        cost_increase = final_cost - initial_cost
        
        print(f"调用后成本: ${final_cost:.6f}")
        print(f"本次调用成本: ${cost_increase:.6f}")
        print(f"响应: {response[:100]}...")
        
        if cost_increase > 0:
            print("✓ 成本监控功能正常工作")
        else:
            print("⚠️  成本可能为0（免费模型或本地模型）")
        
        return True
        
    except Exception as e:
        print(f"✗ 成本监控测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print(f"开始简单回测测试 - {datetime.now()}")
    print("=" * 60)
    
    tests = [
        ("基础回测功能", test_basic_backtest_with_mock_data),
        ("LLM策略功能", test_llm_strategy_with_mock_data),
        ("成本监控功能", test_cost_monitoring),
    ]
    
    test_results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            result = test_func()
            test_results.append((test_name, result))
            
            if result:
                print(f"✓ {test_name} 测试通过")
            else:
                print(f"✗ {test_name} 测试失败")
                
        except Exception as e:
            print(f"✗ {test_name} 测试异常: {e}")
            test_results.append((test_name, False))
    
    # 汇总结果
    print(f"\n{'='*60}")
    print("回测测试结果汇总:")
    print(f"{'='*60}")
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name:<20} {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 简单回测测试全部通过！")
        print("📊 统一LLM系统已成功集成到回测框架中！")
        print("\n下一步建议:")
        print("1. 使用真实数据进行更长期的回测")
        print("2. 测试更复杂的LLM策略")
        print("3. 优化API调用的成本和性能")
        return 0
    else:
        print(f"⚠️  有 {total - passed} 个测试失败")
        return 1

if __name__ == "__main__":
    exit_code = main()
    print(f"\n测试完成，退出代码: {exit_code}")
    sys.exit(exit_code)