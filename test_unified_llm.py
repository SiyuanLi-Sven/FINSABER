#!/usr/bin/env python3
"""
测试统一LLM客户端的功能
包括基础API调用和回测功能验证
"""

import os
import sys
import logging
from datetime import datetime
import traceback

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_config_functions():
    """测试配置功能"""
    print("=== 测试配置功能 ===")
    
    try:
        from config import (
            get_all_llm_models, 
            get_all_embedding_models, 
            get_model_config,
            validate_model_config,
            get_recommended_models
        )
        
        # 测试获取模型列表
        llm_models = get_all_llm_models()
        embedding_models = get_all_embedding_models()
        
        print(f"可用LLM模型: {llm_models}")
        print(f"可用Embedding模型: {embedding_models}")
        
        # 测试推荐模型
        recommended = get_recommended_models()
        print(f"推荐模型: {recommended}")
        
        # 验证模型配置
        for model in llm_models[:2]:  # 只测试前两个
            is_valid = validate_model_config(model)
            print(f"模型 {model} 配置验证: {'✓' if is_valid else '✗'}")
            
        return True
        
    except Exception as e:
        print(f"配置功能测试失败: {e}")
        traceback.print_exc()
        return False

def test_basic_llm_client():
    """测试基础LLM客户端功能"""
    print("\n=== 测试基础LLM客户端 ===")
    
    try:
        from llm_client import create_llm_client, get_available_models
        
        # 使用Qwen/Qwen3-8B进行测试
        test_model = "Qwen/Qwen3-8B"
        print(f"测试模型: {test_model}")
        
        # 创建客户端
        client = create_llm_client(test_model)
        print("✓ LLM客户端创建成功")
        
        # 测试简单对话
        print("测试简单对话...")
        response = client.simple_completion(
            "你好，请简单介绍一下你自己。",
            temperature=0.1,
            max_tokens=100
        )
        print(f"响应: {response[:150]}...")
        print("✓ 简单对话测试成功")
        
        # 测试聊天完成
        print("测试聊天完成...")
        messages = [
            {"role": "system", "content": "你是一个专业的金融分析师。"},
            {"role": "user", "content": "什么是夏普比率？"}
        ]
        response, usage = client.chat_completion(
            messages, 
            temperature=0.2, 
            max_tokens=200
        )
        print(f"聊天响应: {response[:150]}...")
        print(f"使用信息: {usage}")
        print("✓ 聊天完成测试成功")
        
        return True
        
    except Exception as e:
        print(f"LLM客户端测试失败: {e}")
        traceback.print_exc()
        return False

def test_embedding_client():
    """测试Embedding客户端功能"""
    print("\n=== 测试Embedding客户端 ===")
    
    try:
        from llm_client import create_embedding_client
        
        test_model = "Qwen/Qwen3-Embedding-4B"
        print(f"测试Embedding模型: {test_model}")
        
        # 创建客户端
        client = create_embedding_client(test_model)
        print("✓ Embedding客户端创建成功")
        
        # 测试单文本嵌入
        test_text = "这是一个测试文本，用于验证嵌入功能。"
        embedding = client.embed_query(test_text)
        print(f"单文本嵌入维度: {len(embedding)}")
        print("✓ 单文本嵌入测试成功")
        
        # 测试多文本嵌入
        test_texts = [
            "苹果公司股价上涨",
            "市场整体表现良好", 
            "投资需要谨慎考虑"
        ]
        embeddings = client.embed_documents(test_texts)
        print(f"多文本嵌入: {len(embeddings)}个向量，每个维度: {len(embeddings[0])}")
        print("✓ 多文本嵌入测试成功")
        
        return True
        
    except Exception as e:
        print(f"Embedding客户端测试失败: {e}")
        traceback.print_exc()
        return False

def test_finmem_integration():
    """测试FinMem集成"""
    print("\n=== 测试FinMem集成 ===")
    
    try:
        from llm_traders.finmem.puppy.unified_chat import UnifiedChatClient, ChatOpenAICompatible
        
        # 测试新的统一聊天客户端
        print("测试UnifiedChatClient...")
        client = UnifiedChatClient(
            model_name="Qwen/Qwen3-8B",
            system_message="你是一个金融交易助手。",
            temperature=0.1
        )
        
        response = client.simple_chat("当前市场情况如何？请给出简短分析。")
        print(f"FinMem统一客户端响应: {response[:100]}...")
        print("✓ UnifiedChatClient测试成功")
        
        # 测试兼容性客户端
        print("测试兼容性客户端...")
        compat_client = ChatOpenAICompatible(
            model="Qwen/Qwen3-8B",
            system_message="你是一个有用的助手。"
        )
        
        endpoint = compat_client.guardrail_endpoint()
        response = endpoint("请以JSON格式返回市场分析：{'analysis': '你的分析'}")
        print(f"兼容性客户端响应: {response[:100]}...")
        print("✓ 兼容性客户端测试成功")
        
        return True
        
    except Exception as e:
        print(f"FinMem集成测试失败: {e}")
        traceback.print_exc()
        return False

def test_finagent_integration():
    """测试FinAgent集成"""
    print("\n=== 测试FinAgent集成 ===")
    
    try:
        from llm_traders.finagent.provider.unified_provider import UnifiedProvider
        
        # 创建统一Provider
        provider = UnifiedProvider(model_config={
            "llm_model": "Qwen/Qwen3-8B",
            "embedding_model": "Qwen/Qwen3-Embedding-4B"
        })
        print("✓ UnifiedProvider创建成功")
        
        # 测试文本完成
        messages = [
            {"role": "system", "content": "你是一个金融分析专家。"},
            {"role": "user", "content": "请分析一下技术分析的优缺点。"}
        ]
        
        response, info = provider.create_completion(
            messages=messages,
            temperature=0.1,
            max_tokens=200
        )
        
        print(f"FinAgent Provider响应: {response[:100]}...")
        print(f"使用信息: {info}")
        print("✓ FinAgent Provider文本完成测试成功")
        
        # 测试嵌入功能
        test_texts = ["股票分析", "市场趋势", "风险管理"]
        embeddings = provider.embed_documents(test_texts)
        
        if embeddings:
            print(f"嵌入测试成功: {len(embeddings)}个向量，维度: {len(embeddings[0])}")
            print("✓ FinAgent Provider嵌入测试成功")
        else:
            print("✗ 嵌入测试失败")
            return False
        
        return True
        
    except Exception as e:
        print(f"FinAgent集成测试失败: {e}")
        traceback.print_exc()
        return False

def test_simple_backtest():
    """测试简单回测功能"""
    print("\n=== 测试简单回测功能 ===")
    
    try:
        # 导入必要的模块
        from backtest.finsaber import FINSABER
        from backtest.data_util.finmem_dataset import FinMemDataset
        from backtest.strategy.timing.buy_and_hold import BuyAndHoldStrategy
        
        print("✓ 回测模块导入成功")
        
        # 注意：这里不运行实际的回测，因为需要真实数据
        # 仅测试模块导入和基本初始化
        print("回测功能集成正常（跳过实际数据测试）")
        return True
        
    except ImportError as e:
        print(f"回测模块导入失败: {e}")
        return False
    except Exception as e:
        print(f"回测功能测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print(f"开始统一LLM系统测试 - {datetime.now()}")
    print("=" * 60)
    
    test_results = []
    
    # 运行所有测试
    tests = [
        ("配置功能", test_config_functions),
        ("基础LLM客户端", test_basic_llm_client),
        ("Embedding客户端", test_embedding_client),
        ("FinMem集成", test_finmem_integration),
        ("FinAgent集成", test_finagent_integration),
        ("回测功能", test_simple_backtest),
    ]
    
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
            traceback.print_exc()
    
    # 汇总结果
    print(f"\n{'='*60}")
    print("测试结果汇总:")
    print(f"{'='*60}")
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！统一LLM系统重构成功！")
        return 0
    else:
        print(f"⚠️  有 {total - passed} 个测试失败，请检查相关功能")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)