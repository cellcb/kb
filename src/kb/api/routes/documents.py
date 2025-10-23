"""Document Management API Routes"""

import json
import os
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse

from ..models.documents import (
    UploadResponse, DocumentListResponse, DocumentInfo, ProcessingOptions,
    IndexRebuildRequest, ConfigUpdate, DocumentStatus,
    IngestRequest, IngestResponse,
)
from ..dependencies import get_rag_engine, get_task_manager


router = APIRouter()


@router.post("/documents/ingest", response_model=IngestResponse, summary="接收外部文档数据")
async def ingest_documents(request: IngestRequest):
    """接收外部服务提交的已处理文档并更新索引"""
    try:
        rag_engine = get_rag_engine()
        payload = [doc.dict() for doc in request.documents]
        result = await rag_engine.ingest_documents_async(
            payload,
            reset_existing=request.reset_index,
        )

        message = "成功摄取文档" if not request.reset_index else "索引重建完成"
        return IngestResponse(
            success=True,
            indexed_documents=result.get("indexed_documents", 0),
            indexed_nodes=result.get("indexed_nodes", 0),
            keyword_chunks=result.get("keyword_chunks", 0),
            message=message,
        )

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"文档摄取失败: {exc}")


@router.post("/documents/ingest/upload", response_model=IngestResponse, summary="上传文件并索引")
async def ingest_documents_from_files(
    files: List[UploadFile] = File(...),
    reset_index: bool = Form(False),
    metadata: Optional[str] = Form(None, description="可选，JSON格式的文件元数据映射"),
):
    """直接上传PDF/DOCX/TXT文件并写入索引"""
    if not files:
        raise HTTPException(status_code=400, detail="请提供至少一个文件")

    allowed_ext = {'.pdf', '.docx', '.txt'}

    metadata_map: Dict[str, Dict[str, Any]] = {}
    if metadata:
        try:
            raw = json.loads(metadata)
            if not isinstance(raw, dict):
                raise ValueError
            metadata_map = {
                str(key): value
                for key, value in raw.items()
                if isinstance(value, dict)
            }
        except ValueError:
            raise HTTPException(status_code=400, detail="metadata 必须是 JSON 对象，值为字典")

    rag_engine = get_rag_engine()

    uploads: List[Dict[str, Any]] = []
    for upload in files:
        filename = upload.filename or f"upload_{uuid.uuid4().hex[:8]}"
        ext = Path(filename).suffix.lower()
        if ext not in allowed_ext:
            raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")

        data = await upload.read()
        if not data:
            raise HTTPException(status_code=400, detail=f"文件 {filename} 内容为空")

        uploads.append({"filename": filename, "data": data})

    try:
        documents = await rag_engine.documents_from_uploads(uploads)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"文件解析失败: {exc}")

    if not documents:
        raise HTTPException(status_code=400, detail="未能从文件中提取有效文本")

    ingest_payload = []
    for doc in documents:
        doc_metadata = dict(doc.metadata or {})
        filename = doc_metadata.get("filename")
        extra_meta = metadata_map.get(filename, {}) if filename else {}
        combined_metadata = {**doc_metadata, **extra_meta}

        document_id = combined_metadata.pop("document_id", None) or f"doc_{uuid.uuid4().hex[:8]}"

        ingest_payload.append({
            "document_id": document_id,
            "filename": filename,
            "content": doc.text,
            "metadata": combined_metadata,
        })

    try:
        result = await rag_engine.ingest_documents_async(
            ingest_payload,
            reset_existing=reset_index,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"文档摄取失败: {exc}")

    message = "成功摄取文件"
    if reset_index:
        message = "索引重建完成"

    return IngestResponse(
        success=True,
        indexed_documents=result.get("indexed_documents", 0),
        indexed_nodes=result.get("indexed_nodes", 0),
        keyword_chunks=result.get("keyword_chunks", 0),
        message=message,
    )


@router.post("/documents/upload", response_model=UploadResponse, summary="上传文档")
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    parallel_workers: int = Form(4),
    enable_batch_processing: bool = Form(True),
    priority: str = Form("normal")
):
    """
    上传一个或多个文档进行处理
    
    - **files**: 上传的文件列表（支持PDF和TXT格式）
    - **parallel_workers**: 并行工作线程数 (1-16)
    - **enable_batch_processing**: 是否启用批量处理
    - **priority**: 处理优先级 (high/normal/low)
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="请选择要上传的文件")
        
        # 验证并行工作线程数
        if not 1 <= parallel_workers <= 16:
            raise HTTPException(status_code=400, detail="并行工作线程数必须在1-16之间")
        
        # 验证文件格式
        allowed_extensions = {'.pdf', '.txt'}
        uploaded_files = []
        
        for file in files:
            if not file.filename:
                raise HTTPException(status_code=400, detail="文件名不能为空")
            
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in allowed_extensions:
                raise HTTPException(
                    status_code=400, 
                    detail=f"不支持的文件格式: {file_ext}。仅支持: {', '.join(allowed_extensions)}"
                )
            
            # 检查文件大小 (最大100MB)
            if hasattr(file, 'size') and file.size > 100 * 1024 * 1024:
                raise HTTPException(
                    status_code=400,
                    detail=f"文件 {file.filename} 过大，最大支持100MB"
                )
            
            uploaded_files.append(file)
        
        # 保存上传的文件
        rag_engine = get_rag_engine()
        data_dir = rag_engine.data_dir
        data_dir.mkdir(exist_ok=True)
        
        saved_files = []
        document_infos = []
        
        for file in uploaded_files:
            # 生成唯一文件名避免冲突
            file_id = uuid.uuid4().hex[:8]
            original_name = file.filename
            name_parts = Path(original_name).stem, Path(original_name).suffix
            unique_filename = f"{name_parts[0]}_{file_id}{name_parts[1]}"
            
            file_path = data_dir / unique_filename
            
            # 保存文件
            content = await file.read()
            with open(file_path, 'wb') as f:
                f.write(content)
            
            saved_files.append(file_path)
            
            # 创建文档信息
            doc_info = DocumentInfo(
                document_id=f"doc_{file_id}",
                filename=original_name,
                upload_time=datetime.now(),
                file_size=len(content),
                status=DocumentStatus.UPLOADED
            )
            document_infos.append(doc_info)
        
        # 提交处理任务
        task_manager = get_task_manager()
        options = {
            'parallel_workers': parallel_workers,
            'enable_batch_processing': enable_batch_processing,
            'priority': priority
        }
        
        task_id = await task_manager.submit_task(saved_files, options)
        
        # 估算处理时间
        total_size_mb = sum(info.file_size for info in document_infos) / (1024 * 1024)
        if total_size_mb < 1:
            estimated_time = "30秒-1分钟"
        elif total_size_mb < 10:
            estimated_time = "1-3分钟"
        else:
            estimated_time = "3-10分钟"
        
        return UploadResponse(
            success=True,
            message=f"成功上传 {len(files)} 个文件，正在处理中",
            task_id=task_id,
            files=document_infos,
            estimated_time=estimated_time,
            parallel_workers=parallel_workers
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"文件上传失败: {str(e)}"
        )


@router.get("/documents", response_model=DocumentListResponse, summary="获取文档列表")
async def list_documents():
    """
    获取所有已上传文档的列表
    """
    try:
        rag_engine = get_rag_engine()
        data_dir = rag_engine.data_dir
        
        if not data_dir.exists():
            return DocumentListResponse(
                documents=[],
                total_count=0,
                total_size=0
            )
        
        documents = []
        total_size = 0
        
        for file_path in data_dir.iterdir():
            if file_path.is_file():
                file_stat = file_path.stat()
                total_size += file_stat.st_size
                
                # 从缓存中获取字符数信息（如果有的话）
                char_count = None
                if hasattr(rag_engine, 'file_cache'):
                    cache_key = str(file_path)
                    cached_info = rag_engine.file_cache.get(cache_key, {})
                    char_count = cached_info.get('char_count')
                
                doc_info = DocumentInfo(
                    document_id=f"doc_{file_path.stem}",
                    filename=file_path.name,
                    upload_time=datetime.fromtimestamp(file_stat.st_mtime),
                    file_size=file_stat.st_size,
                    status=DocumentStatus.INDEXED,  # 假设已存在的文件都已索引
                    char_count=char_count
                )
                documents.append(doc_info)
        
        # 按上传时间排序
        documents.sort(key=lambda x: x.upload_time, reverse=True)
        
        return DocumentListResponse(
            documents=documents,
            total_count=len(documents),
            total_size=total_size
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取文档列表失败: {str(e)}"
        )


@router.delete("/documents/{document_id}", summary="删除文档")
async def delete_document(document_id: str):
    """
    删除指定文档
    
    - **document_id**: 文档ID
    """
    try:
        rag_engine = get_rag_engine()
        data_dir = rag_engine.data_dir
        
        # 查找对应的文件
        deleted_files = []
        for file_path in data_dir.iterdir():
            if file_path.is_file() and document_id in file_path.stem:
                file_path.unlink()  # 删除文件
                deleted_files.append(file_path.name)
                
                # 从缓存中移除
                if hasattr(rag_engine, 'file_cache'):
                    cache_key = str(file_path)
                    rag_engine.file_cache.pop(cache_key, None)
                    rag_engine._save_file_cache()
        
        if not deleted_files:
            raise HTTPException(
                status_code=404,
                detail=f"文档 {document_id} 不存在"
            )
        
        return {
            "message": f"成功删除文档 {document_id}",
            "deleted_files": deleted_files
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"删除文档失败: {str(e)}"
        )


@router.post("/index/rebuild", summary="重建索引")
async def rebuild_index(request: IndexRebuildRequest):
    """
    重建向量索引
    
    - **parallel_workers**: 并行工作线程数
    - **force**: 是否强制重建
    """
    try:
        rag_engine = get_rag_engine()
        task_manager = get_task_manager()
        
        # 获取所有文档文件
        data_dir = rag_engine.data_dir
        all_files = [f for f in data_dir.iterdir() if f.is_file()]
        
        if not all_files:
            raise HTTPException(
                status_code=400,
                detail="数据目录中没有文件，无法重建索引"
            )
        
        # 如果不是强制重建，检查索引是否已存在
        if not request.force and rag_engine.index is not None:
            return {
                "message": "索引已存在，使用 force=true 强制重建",
                "index_exists": True
            }
        
        # 异步重建索引
        documents = await rag_engine.process_documents_async(all_files)
        await rag_engine.build_index_async(documents, reset_existing=True)
        
        return {
            "message": "索引重建成功",
            "processed_documents": len(documents),
            "parallel_workers": request.parallel_workers
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"重建索引失败: {str(e)}"
        )


@router.put("/config", summary="更新配置")
async def update_config(config: ConfigUpdate):
    """
    更新系统配置
    
    - **max_parallel_workers**: 最大并行工作线程数
    - **embedding_model**: Embedding模型名称
    - **chunk_size**: 文本分块大小
    """
    try:
        rag_engine = get_rag_engine()
        updated_config = {}
        
        # 更新最大并行工作线程数
        if config.max_parallel_workers is not None:
            rag_engine.max_workers = config.max_parallel_workers
            updated_config['max_parallel_workers'] = config.max_parallel_workers
        
        # 更新embedding模型（需要重新初始化）
        if config.embedding_model is not None:
            rag_engine._setup_embedding_model(config.embedding_model)
            updated_config['embedding_model'] = config.embedding_model
        
        # 更新文本分块大小（需要重新初始化node_parser）
        if config.chunk_size is not None:
            from llama_index.core.node_parser import SentenceSplitter
            from llama_index.core import Settings
            Settings.node_parser = SentenceSplitter(
                chunk_size=config.chunk_size, 
                chunk_overlap=20
            )
            updated_config['chunk_size'] = config.chunk_size
        
        return {
            "message": "配置更新成功",
            "updated_config": updated_config
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"更新配置失败: {str(e)}"
        )
