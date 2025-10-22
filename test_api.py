#!/usr/bin/env python3
"""
RAG Demo API 测试脚本
验证API基本功能
"""

import asyncio
import aiohttp
import json
import time
from pathlib import Path

BASE_URL = "http://localhost:8000"

async def test_health():
    """测试健康检查API"""
    print("🏥 测试健康检查API...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/api/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ 健康检查成功: {data}")
                    return True
                else:
                    print(f"❌ 健康检查失败: {response.status}")
                    return False
    except Exception as e:
        print(f"❌ 健康检查连接失败: {e}")
        return False

async def test_status():
    """测试系统状态API"""
    print("📊 测试系统状态API...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/api/status") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ 系统状态: {data}")
                    return True
                else:
                    print(f"❌ 系统状态获取失败: {response.status}")
                    return False
    except Exception as e:
        print(f"❌ 系统状态连接失败: {e}")
        return False

async def test_chat():
    """测试对话API"""
    print("💬 测试对话API...")
    try:
        chat_data = {
            "message": "什么是机器学习？",
            "session_id": "test_session"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{BASE_URL}/api/chat", 
                json=chat_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ 对话测试成功:")
                    print(f"   回答: {data.get('answer', '')[:100]}...")
                    print(f"   来源数量: {len(data.get('sources', []))}")
                    return True
                else:
                    text = await response.text()
                    print(f"❌ 对话测试失败: {response.status} - {text}")
                    return False
    except Exception as e:
        print(f"❌ 对话测试连接失败: {e}")
        return False

async def test_documents_list():
    """测试文档列表API"""
    print("📄 测试文档列表API...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/api/documents") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ 文档列表获取成功:")
                    print(f"   文档数量: {data.get('total_count', 0)}")
                    print(f"   总大小: {data.get('total_size', 0)} 字节")
                    return True
                else:
                    text = await response.text()
                    print(f"❌ 文档列表获取失败: {response.status} - {text}")
                    return False
    except Exception as e:
        print(f"❌ 文档列表连接失败: {e}")
        return False

async def test_upload_text_file():
    """测试文本文件上传"""
    print("⬆️ 测试文本文件上传...")
    
    # 创建一个测试文本文件
    test_file_path = Path("test_upload.txt")
    test_content = """
测试文档内容
这是一个用于测试API上传功能的示例文档。

内容包括：
1. 基本文本信息
2. 测试数据验证
3. API功能确认

该文档将被用于验证RAG系统的文档处理能力。
"""
    
    try:
        # 写入测试文件
        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        # 上传文件
        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field('files', 
                          open(test_file_path, 'rb'),
                          filename='test_upload.txt',
                          content_type='text/plain')
            data.add_field('parallel_workers', '2')
            data.add_field('enable_batch_processing', 'true')
            data.add_field('priority', 'normal')
            
            async with session.post(f"{BASE_URL}/api/documents/upload", data=data) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ 文件上传成功:")
                    print(f"   任务ID: {result.get('task_id')}")
                    print(f"   预估时间: {result.get('estimated_time')}")
                    
                    # 清理测试文件
                    test_file_path.unlink()
                    return True, result.get('task_id')
                else:
                    text = await response.text()
                    print(f"❌ 文件上传失败: {response.status} - {text}")
                    test_file_path.unlink()
                    return False, None
                    
    except Exception as e:
        print(f"❌ 文件上传测试失败: {e}")
        if test_file_path.exists():
            test_file_path.unlink()
        return False, None

async def test_task_status(task_id):
    """测试任务状态API"""
    if not task_id:
        return False
        
    print(f"⏳ 测试任务状态API (任务ID: {task_id})...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/api/tasks/{task_id}") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ 任务状态获取成功:")
                    print(f"   状态: {data.get('status')}")
                    print(f"   进度: {data.get('progress', {}).get('percentage', 0):.1f}%")
                    return True
                else:
                    text = await response.text()
                    print(f"❌ 任务状态获取失败: {response.status} - {text}")
                    return False
    except Exception as e:
        print(f"❌ 任务状态连接失败: {e}")
        return False

async def run_tests():
    """运行所有测试"""
    print("🧪 开始API功能测试...\n")
    
    results = []
    
    # 基础连接测试
    results.append(await test_health())
    await asyncio.sleep(1)
    
    results.append(await test_status())
    await asyncio.sleep(1)
    
    # 文档管理测试
    results.append(await test_documents_list())
    await asyncio.sleep(1)
    
    # 文件上传测试
    upload_success, task_id = await test_upload_text_file()
    results.append(upload_success)
    await asyncio.sleep(2)  # 等待处理开始
    
    # 任务状态测试
    if task_id:
        results.append(await test_task_status(task_id))
        await asyncio.sleep(1)
    
    # 对话功能测试 (需要有文档数据)
    results.append(await test_chat())
    
    # 统计结果
    success_count = sum(results)
    total_count = len(results)
    
    print(f"\n📊 测试结果汇总:")
    print(f"   成功: {success_count}/{total_count}")
    print(f"   成功率: {success_count/total_count*100:.1f}%")
    
    if success_count == total_count:
        print("🎉 所有测试通过！API服务运行正常")
    else:
        print("⚠️  部分测试失败，请检查服务状态")

def main():
    """主函数"""
    print("RAG Demo API 测试工具")
    print("====================")
    print("请确保API服务已启动 (http://localhost:8000)")
    print()
    
    try:
        asyncio.run(run_tests())
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试运行失败: {e}")

if __name__ == "__main__":
    main()
