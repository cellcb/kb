"""
RAG Demo using LlamaIndex
Simple demonstration of Retrieval-Augmented Generation
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
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
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms import CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
import requests


class RAGDemo:
    def __init__(self, data_dir: str = "data", persist_dir: str = "storage", 
                 embedding_model: str = "BAAI/bge-small-zh-v1.5"):
        """初始化RAG演示系统"""
        self.data_dir = Path(data_dir)
        self.persist_dir = Path(persist_dir)
        self.console = Console()
        self.index = None
        
        # 加载环境变量
        # load_dotenv()
        
        # 配置LlamaIndex设置 - 使用自定义DeepSeek LLM
        Settings.llm = OpenAILike(
            model="deepseek-v3-250324",
            api_key="155d5cb5-6b83-4d52-8be8-eb795c72ad44",
            api_base="https://ark.cn-beijing.volces.com/api/v3",
            is_chat_model=True,
            temperature=0.1
        )
        
        # 使用本地embedding模型
        self._setup_embedding_model(embedding_model)
        Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
        
    def _setup_embedding_model(self, model_name: str):
        """配置embedding模型"""
        self.console.print(f"[blue]正在加载embedding模型: {model_name}[/blue]")
        
        # 预定义的模型映射，包含中英文模型
        model_info = {
            "BAAI/bge-small-zh-v1.5": "BGE小型中文模型 (推荐中文使用)",
            "BAAI/bge-base-zh-v1.5": "BGE基础中文模型 (更好效果但更大)",
            "sentence-transformers/all-MiniLM-L6-v2": "轻量级英文模型 (快速)",
            "sentence-transformers/all-mpnet-base-v2": "高质量英文模型",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "多语言模型",
        }
        
        if model_name in model_info:
            self.console.print(f"[green]使用模型: {model_info[model_name]}[/green]")
        
        try:
            Settings.embed_model = HuggingFaceEmbedding(model_name=model_name)
            self.console.print(f"[green]Embedding模型加载成功[/green]")
        except Exception as e:
            self.console.print(f"[red]加载embedding模型失败: {e}[/red]")
            self.console.print("[yellow]尝试使用默认的轻量级模型...[/yellow]")
            Settings.embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        
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
    
    def query(self, question: str) -> Dict[str, Any]:
        """查询RAG系统并返回答案和来源文档"""
        if self.index is None:
            self.get_or_create_index()
        
        query_engine = self.index.as_query_engine(response_mode="compact")
        response = query_engine.query(question)
        
        # 提取来源文档信息
        sources = []
        if hasattr(response, 'source_nodes') and response.source_nodes:
            for node in response.source_nodes:
                source_info = {
                    'filename': node.metadata.get('filename', '未知文档'),
                    'content_preview': node.text[:100] + "..." if len(node.text) > 100 else node.text,
                    'score': getattr(node, 'score', None)
                }
                sources.append(source_info)
        
        return {
            'answer': str(response),
            'sources': sources
        }
    
    def _format_sources(self, sources: List[Dict[str, Any]]) -> str:
        """格式化来源文档信息"""
        if not sources:
            return "\n[dim]未找到参考文档[/dim]"
        
        formatted = "\n[bold yellow]📚 参考文档:[/bold yellow]\n"
        for i, source in enumerate(sources, 1):
            formatted += f"[cyan]{i}. {source['filename']}[/cyan]\n"
            formatted += f"   {source['content_preview']}\n"
            if source.get('score'):
                formatted += f"   [dim]相关度: {source['score']:.3f}[/dim]\n"
        
        return formatted
    
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
                result = self.query(question)
                
                # 显示答案
                self.console.print(Panel(
                    result['answer'],
                    title="[bold green]回答[/bold green]",
                    border_style="green"
                ))
                
                # 显示来源文档
                sources_text = self._format_sources(result['sources'])
                self.console.print(sources_text)
                
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
    parser.add_argument("--embedding-model", default="BAAI/bge-small-zh-v1.5", 
                        help="Embedding模型名称 (默认: BAAI/bge-small-zh-v1.5)")
    parser.add_argument("--list-models", action="store_true", 
                        help="列出推荐的embedding模型")
    parser.add_argument("--test-embedding", action="store_true",
                        help="测试embedding模型（无需API密钥）")
    
    args = parser.parse_args()
    
    # 列出模型选项
    if args.list_models:
        print("\n推荐的Embedding模型:")
        models = {
            "BAAI/bge-small-zh-v1.5": "BGE小型中文模型 (推荐中文使用) - 轻量快速",
            "BAAI/bge-base-zh-v1.5": "BGE基础中文模型 - 更好效果但更大",
            "sentence-transformers/all-MiniLM-L6-v2": "轻量级英文模型 - 快速加载",
            "sentence-transformers/all-mpnet-base-v2": "高质量英文模型 - 最佳效果",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "多语言支持模型",
        }
        for model, desc in models.items():
            print(f"  {model}: {desc}")
        print("\n使用方法: --embedding-model MODEL_NAME")
        return
    
    # 测试embedding模型
    if args.test_embedding:
        test_embedding_model(args.embedding_model)
        return
    
    # 检查OpenAI API密钥
    # if not os.getenv("OPENAI_API_KEY"):
    #     print("错误: 请设置 OPENAI_API_KEY 环境变量")
    #     print("你可以在 .env 文件中设置: OPENAI_API_KEY=your_api_key_here")
    #     return
    
    # 初始化RAG系统
    rag = RAGDemo(
        data_dir=args.data_dir, 
        persist_dir=args.persist_dir,
        embedding_model=args.embedding_model
    )
    
    try:
        if args.rebuild:
            rag.console.print("[yellow]强制重建索引...[/yellow]")
            rag.build_index()
        
        if args.query:
            # 直接查询模式
            result = rag.query(args.query)
            rag.console.print(Panel(
                result['answer'],
                title="[bold green]回答[/bold green]",
                border_style="green"
            ))
            
            # 显示来源文档
            sources_text = rag._format_sources(result['sources'])
            rag.console.print(sources_text)
        else:
            # 交互模式
            rag.interactive_chat()
            
    except Exception as e:
        rag.console.print(f"[red]启动失败: {e}[/red]")


def test_embedding_model(model_name: str):
    """测试embedding模型功能"""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    import numpy as np
    
    console = Console()
    
    console.print(Panel.fit(
        f"[bold blue]测试 Embedding 模型: {model_name}[/bold blue]",
        title="Embedding 测试"
    ))
    
    try:
        # 加载模型
        console.print(f"[blue]正在加载模型: {model_name}[/blue]")
        embed_model = HuggingFaceEmbedding(model_name=model_name)
        console.print("[green]✓ 模型加载成功[/green]")
        
        # 测试文本
        test_texts = [
            "机器学习是人工智能的一个分支",
            "深度学习使用神经网络进行学习",
            "自然语言处理专注于计算机理解人类语言",
            "今天天气很好，适合出门散步",
            "Machine learning is a subset of artificial intelligence"
        ]
        
        console.print("\n[blue]计算文本向量...[/blue]")
        embeddings = []
        
        for i, text in enumerate(test_texts):
            embedding = embed_model.get_text_embedding(text)
            embeddings.append(embedding)
            console.print(f"[green]✓ 文本 {i+1}: 向量维度 {len(embedding)}[/green]")
        
        # 计算相似度
        console.print("\n[blue]计算文本相似度...[/blue]")
        
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        # 创建相似度表格
        table = Table(title="文本相似度矩阵 (前3个AI相关文本)")
        table.add_column("文本", style="cyan", no_wrap=True)
        for i in range(3):
            table.add_column(f"文本{i+1}", justify="center")
        
        ai_texts = test_texts[:3]
        ai_embeddings = embeddings[:3]
        
        for i, text in enumerate(ai_texts):
            row = [f"文本{i+1}: {text[:20]}..."]
            for j in range(3):
                similarity = cosine_similarity(ai_embeddings[i], ai_embeddings[j])
                color = "red" if similarity > 0.8 else "yellow" if similarity > 0.6 else "white"
                row.append(f"[{color}]{similarity:.3f}[/{color}]")
            table.add_row(*row)
        
        console.print(table)
        
        # 显示跨语言相似度
        if len(embeddings) >= 5:
            cn_ml = embeddings[0]  # 中文机器学习
            en_ml = embeddings[4]  # 英文机器学习
            cross_lang_sim = cosine_similarity(cn_ml, en_ml)
            
            console.print(f"\n[bold yellow]跨语言相似度测试:[/bold yellow]")
            console.print(f"中文'机器学习' vs 英文'Machine learning': [bold green]{cross_lang_sim:.3f}[/bold green]")
        
        console.print(f"\n[bold green]✅ Embedding模型 {model_name} 测试完成！[/bold green]")
        
    except Exception as e:
        console.print(f"[red]❌ 测试失败: {e}[/red]")


if __name__ == "__main__":
    main() 