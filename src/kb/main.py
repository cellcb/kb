"""
RAG Demo using LlamaIndex
Simple demonstration of Retrieval-Augmented Generation
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.progress import Progress, TaskID, SpinnerColumn, TimeElapsedColumn
from rich.table import Table

# PDF处理相关导入
import pypdf
import pdfplumber

# 可选的文件类型检测
try:
    import magic
    MAGIC_AVAILABLE = True
except (ImportError, OSError):
    MAGIC_AVAILABLE = False

from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms import CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
import requests

from elasticsearch import Elasticsearch

try:
    from llama_index.vector_stores.elasticsearch import ElasticsearchVectorStore
except ImportError:  # pragma: no cover - fallback for older LlamaIndex版本
    from llama_index.vector_stores.elasticsearch import ElasticsearchStore as ElasticsearchVectorStore


class RAGDemo:
    def __init__(
        self,
        data_dir: str = "data",
        persist_dir: str = "storage",
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        enable_parallel: bool = True,
        max_workers: int = 2,
        es_url: Optional[str] = None,
        es_index: str = "kb-documents",
        es_user: Optional[str] = None,
        es_password: Optional[str] = None,
    ):
        """初始化RAG演示系统"""
        self.data_dir = Path(data_dir)
        self.persist_dir = Path(persist_dir)
        self.console = Console()
        self.index = None
        
        # 性能优化配置
        self.enable_parallel = enable_parallel
        self.max_workers = max_workers
        self.cache_dir = self.persist_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # 文件处理缓存
        self.file_cache_path = self.cache_dir / "file_cache.json"
        self.file_cache = self._load_file_cache()
        self.vector_store = None
        self.storage_context = None
        self.es_client = None
        
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
        
        # Elasticsearch 配置
        self.es_url = es_url or os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
        self.es_index = os.getenv("ELASTICSEARCH_INDEX", es_index)
        self.es_user = es_user or os.getenv("ELASTICSEARCH_USER")
        self.es_password = es_password or os.getenv("ELASTICSEARCH_PASSWORD")
        self.es_api_key = os.getenv("ELASTICSEARCH_API_KEY")
        verify_env = os.getenv("ELASTICSEARCH_VERIFY_CERTS")
        self.es_verify_certs: Optional[bool] = None
        if verify_env is not None:
            self.es_verify_certs = verify_env.lower() not in {"false", "0", "no"}
        self.es_ca_certs = os.getenv("ELASTICSEARCH_CA_CERTS")
        timeout_env = os.getenv("ELASTICSEARCH_TIMEOUT")
        self.es_timeout = int(timeout_env) if timeout_env and timeout_env.isdigit() else None
        
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
    
    def _init_vector_store(self, force: bool = False):
        """初始化或重建 Elasticsearch 向量存储"""
        if self.vector_store is not None and self.storage_context is not None and not force:
            return
        
        es_kwargs = {}
        if self.es_user and self.es_password:
            es_kwargs["basic_auth"] = (self.es_user, self.es_password)
        if self.es_api_key:
            es_kwargs["api_key"] = self.es_api_key
        if self.es_verify_certs is not None:
            es_kwargs["verify_certs"] = self.es_verify_certs
        if self.es_ca_certs:
            es_kwargs["ca_certs"] = self.es_ca_certs
        if self.es_timeout:
            es_kwargs["request_timeout"] = self.es_timeout
        
        try:
            self.es_client = Elasticsearch(self.es_url, **es_kwargs)
            self.console.print(f"[blue]连接 Elasticsearch: {self.es_url}[/blue]")
        except Exception as exc:
            self.console.print(f"[red]连接 Elasticsearch 失败: {exc}[/red]")
            raise
        
        try:
            self.es_client.ping()
        except Exception as exc:  # pragma: no cover - ping 失败不阻断
            self.console.print(f"[yellow]Elasticsearch ping 失败: {exc}[/yellow]")
        try:
            self.vector_store = ElasticsearchVectorStore(
                index_name=self.es_index,
                es_url=self.es_url,
                es_user=self.es_user,
                es_password=self.es_password,
                es_api_key=self.es_api_key,
            )
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            self.console.print(f"[green]Elasticsearch 向量索引就绪: {self.es_index}[/green]")
        except Exception as exc:
            self.console.print(f"[red]初始化 Elasticsearch 向量存储失败: {exc}[/red]")
            raise
    
    def _ensure_vector_store(self):
        """确保向量存储已初始化"""
        if self.vector_store is None or self.storage_context is None:
            self._init_vector_store()
    
    def _reset_vector_index(self):
        """删除并重建 Elasticsearch 索引"""
        self._ensure_vector_store()
        
        try:
            delete_method = getattr(self.vector_store, "delete_index", None)
            if callable(delete_method):
                delete_method()
                self.console.print(f"[yellow]已清空 Elasticsearch 索引 {self.es_index}[/yellow]")
            else:
                self.es_client.indices.delete(index=self.es_index, ignore_unavailable=True)
                self.console.print(f"[yellow]已删除 Elasticsearch 索引 {self.es_index}[/yellow]")
        except Exception as exc:
            self.console.print(f"[yellow]删除索引时遇到问题: {exc}[/yellow]")
        finally:
            self._init_vector_store(force=True)
    
    def _refresh_vector_index(self):
        """刷新 Elasticsearch 索引"""
        if not self.es_client:
            return
        try:
            self.es_client.indices.refresh(index=self.es_index)
        except Exception as exc:
            self.console.print(f"[yellow]刷新 Elasticsearch 索引失败: {exc}[/yellow]")
    
    def _load_file_cache(self) -> Dict[str, Any]:
        """加载文件处理缓存"""
        try:
            if self.file_cache_path.exists():
                with open(self.file_cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            self.console.print(f"[yellow]缓存加载失败: {e}[/yellow]")
        return {}
    
    def _save_file_cache(self):
        """保存文件处理缓存"""
        try:
            with open(self.file_cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.file_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.console.print(f"[yellow]缓存保存失败: {e}[/yellow]")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """计算文件的哈希值用于缓存判断"""
        try:
            stat = file_path.stat()
            # 使用文件路径、大小、修改时间创建哈希
            content = f"{file_path.name}_{stat.st_size}_{stat.st_mtime}"
            return hashlib.md5(content.encode()).hexdigest()
        except Exception:
            return ""
    
    def _is_file_cached(self, file_path: Path) -> bool:
        """检查文件是否已被缓存且未过期"""
        file_hash = self._get_file_hash(file_path)
        if not file_hash:
            return False
        
        cache_key = str(file_path)
        if cache_key in self.file_cache:
            cached_info = self.file_cache[cache_key]
            return cached_info.get('hash') == file_hash
        return False
    
    def _get_cached_content(self, file_path: Path) -> Optional[str]:
        """获取缓存的文件内容"""
        cache_key = str(file_path)
        if cache_key in self.file_cache:
            return self.file_cache[cache_key].get('content')
        return None
    
    def _cache_file_content(self, file_path: Path, content: str):
        """缓存文件内容"""
        file_hash = self._get_file_hash(file_path)
        if file_hash:
            cache_key = str(file_path)
            self.file_cache[cache_key] = {
                'hash': file_hash,
                'content': content,
                'timestamp': time.time(),
                'char_count': len(content)
            }
        
    def _detect_file_type(self, file_path: Path) -> str:
        """检测文件的真实类型（基于文件内容，不是扩展名）"""
        if MAGIC_AVAILABLE:
            try:
                mime_type = magic.from_file(str(file_path), mime=True)
                
                if mime_type == 'application/pdf':
                    return 'pdf'
                elif mime_type.startswith('text/'):
                    return 'txt'
                else:
                    # 降级到扩展名检测
                    return file_path.suffix.lower().lstrip('.')
            except Exception:
                # 如果magic检测失败，使用扩展名
                return file_path.suffix.lower().lstrip('.')
        else:
            # magic不可用，直接使用扩展名检测
            return file_path.suffix.lower().lstrip('.')
    
    def _extract_pdf_content(self, file_path: Path) -> Optional[str]:
        """从PDF文件提取文本内容，包含缓存、错误处理和进度显示"""
        
        # 首先检查缓存
        if self._is_file_cached(file_path):
            cached_content = self._get_cached_content(file_path)
            if cached_content:
                self.console.print(f"[green]✓ 使用缓存内容: {file_path.name} ({len(cached_content)} 字符)[/green]")
                return cached_content
        
        # 检查文件大小和基本有效性
        try:
            file_size = file_path.stat().st_size
            if file_size == 0:
                self.console.print(f"[yellow]跳过空文件: {file_path.name}[/yellow]")
                return None
                
            # 大文件警告
            if file_size > 50 * 1024 * 1024:  # 50MB
                self.console.print(f"[yellow]警告: 大文件 {file_path.name} ({file_size / 1024 / 1024:.1f}MB)，处理可能较慢[/yellow]")
                
        except OSError as e:
            self.console.print(f"[red]无法访问文件 {file_path.name}: {e}[/red]")
            return None
        
        try:
            # 首先尝试使用pypdf（速度较快）
            with self.console.status(f"[blue]使用pypdf提取 {file_path.name}..."):
                text = self._extract_with_pypdf(file_path)
                
            if text and text.strip():
                self.console.print(f"[green]✓ pypdf成功提取 {len(text)} 字符[/green]")
                # 缓存成功提取的内容
                self._cache_file_content(file_path, text)
                return text
            
            # 如果pypdf失败或结果为空，尝试pdfplumber
            self.console.print(f"[yellow]pypdf提取结果为空，尝试pdfplumber: {file_path.name}[/yellow]")
            
            with self.console.status(f"[blue]使用pdfplumber提取 {file_path.name}..."):
                text = self._extract_with_pdfplumber(file_path)
                
            if text and text.strip():
                self.console.print(f"[green]✓ pdfplumber成功提取 {len(text)} 字符[/green]")
                # 缓存成功提取的内容
                self._cache_file_content(file_path, text)
                return text
            else:
                self.console.print(f"[yellow]⚠ PDF文件 {file_path.name} 可能是扫描版或损坏[/yellow]")
                return None
            
        except MemoryError:
            self.console.print(f"[red]❌ 内存不足，无法处理大文件: {file_path.name}[/red]")
            return None
        except PermissionError:
            self.console.print(f"[red]❌ 权限不足，无法读取文件: {file_path.name}[/red]")
            return None
        except Exception as e:
            error_type = type(e).__name__
            self.console.print(f"[red]❌ PDF提取失败 {file_path.name} ({error_type}): {str(e)[:100]}[/red]")
            return None
    
    def _extract_with_pypdf(self, file_path: Path) -> str:
        """使用pypdf提取PDF文本，包含页面级进度显示"""
        text = ""
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                if total_pages == 0:
                    self.console.print(f"[yellow]PDF文件 {file_path.name} 没有页面[/yellow]")
                    return ""
                
                # 对于多页PDF显示进度
                if total_pages > 5:
                    with Progress(
                        SpinnerColumn(),
                        "[progress.description]{task.description}",
                        "[progress.percentage]{task.percentage:>3.0f}%",
                        TimeElapsedColumn(),
                        console=self.console
                    ) as progress:
                        task = progress.add_task(f"提取 {file_path.name}", total=total_pages)
                        
                        for page_num, page in enumerate(pdf_reader.pages):
                            try:
                                page_text = page.extract_text()
                                if page_text and page_text.strip():
                                    text += f"\n--- 第{page_num + 1}页 ---\n{page_text}\n"
                                progress.advance(task)
                            except Exception as e:
                                self.console.print(f"[yellow]跳过第{page_num + 1}页: {str(e)[:50]}[/yellow]")
                                progress.advance(task)
                                continue
                else:
                    # 少页数PDF直接处理
                    for page_num, page in enumerate(pdf_reader.pages):
                        try:
                            page_text = page.extract_text()
                            if page_text and page_text.strip():
                                text += f"\n--- 第{page_num + 1}页 ---\n{page_text}\n"
                        except Exception as e:
                            self.console.print(f"[yellow]跳过第{page_num + 1}页: {str(e)[:50]}[/yellow]")
                            continue
                            
        except Exception as e:
            raise Exception(f"pypdf读取失败: {e}")
            
        return text.strip()
    
    def _extract_with_pdfplumber(self, file_path: Path) -> str:
        """使用pdfplumber提取PDF文本（处理复杂布局），包含进度显示"""
        text = ""
        
        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                
                if total_pages == 0:
                    self.console.print(f"[yellow]PDF文件 {file_path.name} 没有页面[/yellow]")
                    return ""
                
                # 对于多页PDF显示进度
                if total_pages > 3:
                    with Progress(
                        SpinnerColumn(),
                        "[progress.description]{task.description}",
                        "[progress.percentage]{task.percentage:>3.0f}%",
                        TimeElapsedColumn(),
                        console=self.console
                    ) as progress:
                        task = progress.add_task(f"pdfplumber处理 {file_path.name}", total=total_pages)
                        
                        for page_num, page in enumerate(pdf.pages):
                            try:
                                page_text = page.extract_text()
                                if page_text and page_text.strip():
                                    text += f"\n--- 第{page_num + 1}页 ---\n{page_text}\n"
                                progress.advance(task)
                            except Exception as e:
                                self.console.print(f"[yellow]跳过第{page_num + 1}页: {str(e)[:50]}[/yellow]")
                                progress.advance(task)
                                continue
                else:
                    # 少页数PDF直接处理
                    for page_num, page in enumerate(pdf.pages):
                        try:
                            page_text = page.extract_text()
                            if page_text and page_text.strip():
                                text += f"\n--- 第{page_num + 1}页 ---\n{page_text}\n"
                        except Exception as e:
                            self.console.print(f"[yellow]跳过第{page_num + 1}页: {str(e)[:50]}[/yellow]")
                            continue
                            
        except Exception as e:
            raise Exception(f"pdfplumber处理失败: {e}")
            
        return text.strip()
    
    def _process_single_file(self, file_path: Path) -> Optional[Document]:
        """处理单个文件，返回Document对象或None"""
        detected_type = self._detect_file_type(file_path)
        
        try:
            if detected_type == 'pdf':
                content = self._extract_pdf_content(file_path)
                if content:
                    return Document(
                        text=content, 
                        metadata={
                            "filename": file_path.name,
                            "file_type": "pdf",
                            "file_path": str(file_path),
                            "file_size": file_path.stat().st_size,
                            "char_count": len(content)
                        }
                    )
                    
            elif detected_type == 'txt':
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if content.strip():
                            return Document(
                                text=content, 
                                metadata={
                                    "filename": file_path.name,
                                    "file_type": "txt",
                                    "file_path": str(file_path),
                                    "file_size": file_path.stat().st_size,
                                    "char_count": len(content)
                                }
                            )
                except UnicodeDecodeError:
                    self.console.print(f"[yellow]文本文件编码错误，跳过: {file_path.name}[/yellow]")
                    
        except Exception as e:
            self.console.print(f"[red]处理文件失败 {file_path.name}: {str(e)[:100]}[/red]")
        
        return None
        
    def load_documents(self) -> List[Document]:
        """从数据目录加载文档（支持txt和pdf格式），包含详细进度和统计"""
        documents = []
        
        if not self.data_dir.exists():
            self.console.print(f"[yellow]数据目录 {self.data_dir} 不存在，创建示例文档...[/yellow]")
            self._create_sample_documents()
        
        # 首先扫描所有文件获取总数
        all_files = [f for f in self.data_dir.iterdir() if f.is_file()]
        if not all_files:
            self.console.print("[yellow]数据目录中没有找到文件[/yellow]")
            return documents
        
        self.console.print(f"[blue]发现 {len(all_files)} 个文件，开始处理...[/blue]")
        
        # 统计信息
        stats = {
            'processed': 0,
            'skipped': 0,
            'pdf_files': 0,
            'txt_files': 0,
            'other_files': 0,
            'total_chars': 0,
            'failed': 0
        }
        
        # 处理文件 - 支持并行或串行
        start_time = time.time()
        
        if self.enable_parallel and len(all_files) > 1:
            documents = self._process_files_parallel(all_files, stats)
        else:
            documents = self._process_files_sequential(all_files, stats)
        
        # 保存缓存
        self._save_file_cache()
        
        processing_time = time.time() - start_time
        
        # 显示详细统计信息
        self._display_processing_stats(stats, processing_time)
        return documents
    
    def _process_files_sequential(self, all_files: List[Path], stats: Dict[str, int]) -> List[Document]:
        """串行处理文件"""
        documents = []
        
        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            "[progress.percentage]{task.percentage:>3.0f}%",
            "[progress.completed]{task.completed}/{task.total}",
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            main_task = progress.add_task("串行处理文档", total=len(all_files))
            
            for file_path in all_files:
                progress.update(main_task, description=f"处理 {file_path.name}")
                
                doc = self._process_single_file(file_path)
                if doc:
                    documents.append(doc)
                    file_type = doc.metadata.get('file_type', 'unknown')
                    if file_type == 'pdf':
                        stats['pdf_files'] += 1
                    elif file_type == 'txt':
                        stats['txt_files'] += 1
                    
                    stats['processed'] += 1
                    stats['total_chars'] += doc.metadata.get('char_count', 0)
                else:
                    # 确定跳过原因
                    detected_type = self._detect_file_type(file_path)
                    if detected_type in ['pdf', 'txt']:
                        stats['failed'] += 1
                    else:
                        stats['other_files'] += 1
                        stats['skipped'] += 1
                
                progress.advance(main_task)
        
        return documents
    
    def _process_files_parallel(self, all_files: List[Path], stats: Dict[str, int]) -> List[Document]:
        """并行处理文件"""
        documents = []
        
        self.console.print(f"[blue]使用并行处理 (最大 {self.max_workers} 个工作线程)[/blue]")
        
        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            "[progress.percentage]{task.percentage:>3.0f}%",
            "[progress.completed]{task.completed}/{task.total}",
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            main_task = progress.add_task("并行处理文档", total=len(all_files))
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有任务
                future_to_file = {
                    executor.submit(self._process_single_file, file_path): file_path 
                    for file_path in all_files
                }
                
                # 收集结果
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    progress.update(main_task, description=f"完成 {file_path.name}")
                    
                    try:
                        doc = future.result()
                        if doc:
                            documents.append(doc)
                            file_type = doc.metadata.get('file_type', 'unknown')
                            if file_type == 'pdf':
                                stats['pdf_files'] += 1
                            elif file_type == 'txt':
                                stats['txt_files'] += 1
                            
                            stats['processed'] += 1
                            stats['total_chars'] += doc.metadata.get('char_count', 0)
                        else:
                            # 确定跳过原因
                            detected_type = self._detect_file_type(file_path)
                            if detected_type in ['pdf', 'txt']:
                                stats['failed'] += 1
                            else:
                                stats['other_files'] += 1
                                stats['skipped'] += 1
                                
                    except Exception as e:
                        self.console.print(f"[red]并行处理失败 {file_path.name}: {e}[/red]")
                        stats['failed'] += 1
                    
                    progress.advance(main_task)
        
        return documents
    
    def _display_processing_stats(self, stats: Dict[str, int], processing_time: float):
        """显示文档处理统计信息，包含性能数据"""
        table = Table(title="文档处理统计", show_header=True, header_style="bold magenta")
        table.add_column("项目", style="cyan", no_wrap=True)
        table.add_column("数量", justify="right", style="green")
        table.add_column("说明", style="dim")
        
        table.add_row("成功处理", str(stats['processed']), "✓ 已加载到向量数据库")
        table.add_row("PDF文件", str(stats['pdf_files']), "通过pypdf/pdfplumber提取")
        table.add_row("文本文件", str(stats['txt_files']), "直接读取")
        table.add_row("跳过文件", str(stats['skipped']), "空文件或不支持格式")
        table.add_row("失败文件", str(stats['failed']), "❌ 处理过程中出错")
        table.add_row("总字符数", f"{stats['total_chars']:,}", "提取的文本总长度")
        table.add_row("处理时间", f"{processing_time:.2f}秒", f"{'并行' if self.enable_parallel else '串行'}处理")
        
        self.console.print(table)
        
        # 性能指标
        if stats['processed'] > 0:
            avg_chars = stats['total_chars'] // stats['processed']
            chars_per_sec = stats['total_chars'] / processing_time if processing_time > 0 else 0
            files_per_sec = stats['processed'] / processing_time if processing_time > 0 else 0
            
            self.console.print(f"[green]✅ 平均每个文档 {avg_chars:,} 字符[/green]")
            self.console.print(f"[blue]📊 处理速度: {files_per_sec:.1f} 文件/秒, {chars_per_sec:,.0f} 字符/秒[/blue]")
        
        # 缓存统计
        cached_files = sum(1 for v in self.file_cache.values() if 'content' in v)
        if cached_files > 0:
            self.console.print(f"[cyan]💾 缓存中有 {cached_files} 个文件，下次处理将更快[/cyan]")
        
        if stats['failed'] > 0:
            self.console.print(f"[yellow]⚠️  {stats['failed']} 个文件处理失败，请检查文件完整性[/yellow]")
    
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
    
    def build_index(self, reset_existing: bool = False) -> VectorStoreIndex:
        """构建向量索引（写入 Elasticsearch）"""
        self.console.print("[blue]构建向量索引（Elasticsearch）...[/blue]")
        self._ensure_vector_store()
        
        documents = self.load_documents()
        if not documents:
            raise ValueError("没有找到要索引的文档")
        
        if reset_existing:
            self._reset_vector_index()
        
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            show_progress=True,
        )
        self._refresh_vector_index()
        
        self.console.print("[green]Elasticsearch 向量索引构建完成[/green]")
        self.index = index
        return index
    
    def load_index(self) -> Optional[VectorStoreIndex]:
        """尝试加载现有的 Elasticsearch 向量索引"""
        self._ensure_vector_store()
        
        try:
            if not self.es_client.indices.exists(index=self.es_index):
                self.console.print("[yellow]Elasticsearch 索引不存在，需要重新构建[/yellow]")
                return None
        except Exception as exc:
            self.console.print(f"[yellow]检测 Elasticsearch 索引失败: {exc}[/yellow]")
            return None
        
        try:
            index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                storage_context=self.storage_context,
            )
            self.console.print("[green]成功连接到 Elasticsearch 向量索引[/green]")
            self.index = index
            return index
        except Exception as exc:
            self.console.print(f"[yellow]加载 Elasticsearch 索引失败: {exc}[/yellow]")
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
        self._ensure_vector_store()
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
    parser.add_argument("--disable-parallel", action="store_true",
                        help="禁用并行处理，使用串行模式")
    parser.add_argument("--max-workers", type=int, default=2,
                        help="并行处理的最大工作线程数 (默认: 2)")
    parser.add_argument("--es-url", help="Elasticsearch 地址 (默认读取环境变量 ELASTICSEARCH_URL)")
    parser.add_argument("--es-index", default="kb-documents",
                        help="Elasticsearch 索引名称 (默认: kb-documents)")
    parser.add_argument("--es-user", help="Elasticsearch 基本认证用户名")
    parser.add_argument("--es-password", help="Elasticsearch 基本认证密码")
    
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
        embedding_model=args.embedding_model,
        enable_parallel=not args.disable_parallel,
        max_workers=args.max_workers,
        es_url=args.es_url,
        es_index=args.es_index,
        es_user=args.es_user,
        es_password=args.es_password,
    )
    
    try:
        if args.rebuild:
            rag.console.print("[yellow]强制重建索引...[/yellow]")
            rag.build_index(reset_existing=True)
        
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
