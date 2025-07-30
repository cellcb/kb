"""
RAG Demo using LlamaIndex
Simple demonstration of Retrieval-Augmented Generation
"""

import os
from pathlib import Path
from typing import List, Optional
import argparse

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding


class RAGDemo:
    def __init__(self, data_dir: str = "data", persist_dir: str = "storage"):
        """初始化RAG演示系统"""
        self.data_dir = Path(data_dir)
        self.persist_dir = Path(persist_dir)
        self.console = Console()
        self.index = None
        
        # 加载环境变量
        load_dotenv()
        
        # 配置LlamaIndex设置
        Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
        Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
        
    def load_documents(self) -> List[Document]:
        """从数据目录加载文档"""
        documents = []
        
        if not self.data_dir.exists():
            self.console.print(f"[yellow]数据目录 {self.data_dir} 不存在，创建示例文档...[/yellow]")
            self._create_sample_documents()
        
        for file_path in self.data_dir.glob("*.txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                doc = Document(text=content, metadata={"filename": file_path.name})
                documents.append(doc)
                
        self.console.print(f"[green]加载了 {len(documents)} 个文档[/green]")
        return documents
    
    def _create_sample_documents(self):
        """创建示例文档"""
        self.data_dir.mkdir(exist_ok=True)
        
        sample_docs = {
            "machine_learning.txt": """
机器学习是人工智能的一个分支，它使计算机系统能够通过经验自动改进其性能。
机器学习算法构建数学模型，基于训练数据进行预测或决策，而无需明确编程。

主要类型包括：
1. 监督学习：使用标记的训练数据学习输入和输出之间的映射
2. 无监督学习：从未标记的数据中发现隐藏的模式
3. 强化学习：通过与环境交互来学习最优行为

常见算法包括线性回归、逻辑回归、决策树、随机森林、支持向量机、神经网络等。
            """,
            "deep_learning.txt": """
深度学习是机器学习的一个子集，使用具有多个隐藏层的人工神经网络。
这些深层网络能够学习数据的复杂表示，在图像识别、自然语言处理等领域表现出色。

关键概念：
- 神经网络：由相互连接的节点（神经元）组成的网络
- 反向传播：用于训练神经网络的算法
- 卷积神经网络（CNN）：专门用于处理图像数据
- 循环神经网络（RNN）：适合处理序列数据
- 变换器（Transformer）：现代NLP的基础架构

深度学习在计算机视觉、语音识别、机器翻译等领域取得了突破性进展。
            """,
            "natural_language_processing.txt": """
自然语言处理（NLP）是人工智能和语言学的交叉领域，专注于计算机理解和生成人类语言。

主要任务包括：
- 文本分类：将文本分配到预定义的类别
- 命名实体识别：识别文本中的人名、地名、组织名等
- 情感分析：确定文本的情感倾向
- 机器翻译：将文本从一种语言翻译成另一种语言
- 问答系统：根据问题从文本中提取答案
- 文本摘要：生成文本的简洁摘要

现代NLP广泛使用Transformer架构，如BERT、GPT等大型语言模型。
这些模型通过在大规模文本数据上预训练，然后在特定任务上微调来实现优异性能。
            """
        }
        
        for filename, content in sample_docs.items():
            file_path = self.data_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content.strip())
        
        self.console.print(f"[green]创建了 {len(sample_docs)} 个示例文档[/green]")
    
    def build_index(self) -> VectorStoreIndex:
        """构建向量索引"""
        self.console.print("[blue]构建向量索引...[/blue]")
        
        documents = self.load_documents()
        if not documents:
            raise ValueError("没有找到要索引的文档")
        
        # 创建索引
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        
        # 持久化索引
        self.persist_dir.mkdir(exist_ok=True)
        index.storage_context.persist(persist_dir=str(self.persist_dir))
        
        self.console.print("[green]索引构建完成并已保存[/green]")
        return index
    
    def load_index(self) -> Optional[VectorStoreIndex]:
        """加载已存在的索引"""
        if not self.persist_dir.exists():
            return None
            
        try:
            storage_context = StorageContext.from_defaults(persist_dir=str(self.persist_dir))
            index = load_index_from_storage(storage_context)
            self.console.print("[green]成功加载已存在的索引[/green]")
            return index
        except Exception as e:
            self.console.print(f"[yellow]加载索引失败: {e}[/yellow]")
            return None
    
    def get_or_create_index(self) -> VectorStoreIndex:
        """获取或创建索引"""
        index = self.load_index()
        if index is None:
            index = self.build_index()
        
        self.index = index
        return index
    
    def query(self, question: str) -> str:
        """查询RAG系统"""
        if self.index is None:
            self.get_or_create_index()
        
        query_engine = self.index.as_query_engine(response_mode="compact")
        response = query_engine.query(question)
        return str(response)
    
    def interactive_chat(self):
        """交互式聊天界面"""
        self.console.print(Panel.fit(
            "[bold blue]RAG Demo 交互式问答系统[/bold blue]\n"
            "输入问题来查询知识库，输入 'exit' 或 'quit' 退出",
            title="欢迎使用RAG Demo"
        ))
        
        # 确保索引已加载
        self.get_or_create_index()
        
        while True:
            try:
                question = Prompt.ask("\n[bold cyan]请输入您的问题[/bold cyan]")
                
                if question.lower() in ['exit', 'quit', '退出']:
                    self.console.print("[yellow]再见！[/yellow]")
                    break
                
                if not question.strip():
                    continue
                
                self.console.print("[blue]正在搜索答案...[/blue]")
                answer = self.query(question)
                
                self.console.print(Panel(
                    answer,
                    title="[bold green]回答[/bold green]",
                    border_style="green"
                ))
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]再见！[/yellow]")
                break
            except Exception as e:
                self.console.print(f"[red]发生错误: {e}[/red]")


def main():
    parser = argparse.ArgumentParser(description="RAG Demo using LlamaIndex")
    parser.add_argument("--data-dir", default="data", help="文档数据目录")
    parser.add_argument("--persist-dir", default="storage", help="索引存储目录")
    parser.add_argument("--query", help="直接查询而不进入交互模式")
    parser.add_argument("--rebuild", action="store_true", help="强制重建索引")
    
    args = parser.parse_args()
    
    # 检查OpenAI API密钥
    if not os.getenv("OPENAI_API_KEY"):
        print("错误: 请设置 OPENAI_API_KEY 环境变量")
        print("你可以在 .env 文件中设置: OPENAI_API_KEY=your_api_key_here")
        return
    
    # 初始化RAG系统
    rag = RAGDemo(data_dir=args.data_dir, persist_dir=args.persist_dir)
    
    try:
        if args.rebuild:
            rag.console.print("[yellow]强制重建索引...[/yellow]")
            rag.build_index()
        
        if args.query:
            # 直接查询模式
            answer = rag.query(args.query)
            rag.console.print(Panel(
                answer,
                title="[bold green]回答[/bold green]",
                border_style="green"
            ))
        else:
            # 交互模式
            rag.interactive_chat()
            
    except Exception as e:
        rag.console.print(f"[red]启动失败: {e}[/red]")


if __name__ == "__main__":
    main() 