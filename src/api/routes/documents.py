"""Document Management API Routes"""

import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from services.task_manager import QueuedDocument
from ..dependencies import get_knowledge_service, get_task_manager, get_tenant_id
from ..models.documents import (
    ConfigUpdate,
    DocumentInfo,
    DocumentListResponse,
    DocumentStatus,
    IndexRebuildRequest,
    IngestRequest,
    IngestResponse,
    ProcessingOptions,
    UploadResponse,
)


router = APIRouter()


@router.post("/documents/ingest", response_model=IngestResponse, summary="接收外部文档数据")
async def ingest_documents(request: IngestRequest, tenant_id: str = Depends(get_tenant_id)):
    """接收外部服务提交的已处理文档并更新索引"""
    try:
        knowledge_service = get_knowledge_service()
        payload = [doc.dict() for doc in request.documents]
        result = await knowledge_service.ingest_documents_async(
            payload,
            reset_existing=request.reset_index,
            tenant_id=tenant_id,
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
    tenant_id: str = Depends(get_tenant_id),
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

    knowledge_service = get_knowledge_service()

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
        documents = await knowledge_service.documents_from_uploads(uploads)
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
        result = await knowledge_service.ingest_documents_async(
            ingest_payload,
            reset_existing=reset_index,
            tenant_id=tenant_id,
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


DOCUMENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def _metadata_path_for(file_path: Path) -> Path:
    """返回与文件对应的元数据路径"""
    return file_path.with_suffix(file_path.suffix + ".meta.json")


def _load_document_metadata(file_path: Path) -> Dict[str, Any]:
    """读取上传文件的元数据，如不存在则返回空"""
    meta_path = _metadata_path_for(file_path)
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as meta_file:
            data = json.load(meta_file)
            if isinstance(data, dict):
                return data
    except (OSError, json.JSONDecodeError):
        pass
    return {}


@router.post("/documents/upload", response_model=UploadResponse, summary="上传文档")
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    parallel_workers: int = Form(4),
    enable_batch_processing: bool = Form(True),
    priority: str = Form("normal"),
    document_ids: Optional[str] = Form(
        None,
        description="可选，自定义文档ID。支持 JSON 数组（按文件顺序）或 JSON 对象（filename -> document_id）。",
    ),
    tenant_id: str = Depends(get_tenant_id),
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
        document_id_map: Dict[str, str] = {}
        document_id_sequence: List[str] = []
        sequence_index = 0

        if document_ids:
            try:
                parsed_ids = json.loads(document_ids)
            except json.JSONDecodeError:
                parsed_ids = document_ids.strip()

            if isinstance(parsed_ids, dict):
                for key, value in parsed_ids.items():
                    if not isinstance(value, str) or not value.strip():
                        raise HTTPException(
                            status_code=400,
                            detail="document_ids 中的值必须为非空字符串",
                        )
                    document_id_map[str(key)] = value.strip()
            elif isinstance(parsed_ids, list):
                cleaned = []
                for item in parsed_ids:
                    if not isinstance(item, str) or not item.strip():
                        raise HTTPException(
                            status_code=400,
                            detail="document_ids 数组元素必须为非空字符串",
                        )
                    cleaned.append(item.strip())
                document_id_sequence = cleaned
            elif isinstance(parsed_ids, str):
                if parsed_ids:
                    document_id_sequence = [parsed_ids]
            else:
                raise HTTPException(
                    status_code=400,
                    detail="document_ids 仅支持 JSON 对象或数组",
                )

        queued_documents: List[QueuedDocument] = []

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
        knowledge_service = get_knowledge_service()
        data_dir = knowledge_service.data_dir
        data_dir.mkdir(exist_ok=True)
        
        document_infos = []
        for file in uploaded_files:
            original_name = file.filename
            file_id = uuid.uuid4().hex[:8]
            provided_id = document_id_map.get(original_name)
            if provided_id is None and sequence_index < len(document_id_sequence):
                provided_id = document_id_sequence[sequence_index]
                sequence_index += 1

            if provided_id:
                document_id = provided_id.strip()
                if not DOCUMENT_ID_PATTERN.fullmatch(document_id):
                    raise HTTPException(
                        status_code=400,
                        detail=f"document_id '{document_id}' 仅支持字母、数字、下划线、点和连字符",
                    )
            else:
                document_id = f"doc_{file_id}"

            name_parts = Path(original_name).stem, Path(original_name).suffix
            unique_filename = f"{name_parts[0]}_{file_id}{name_parts[1]}"
            file_path = data_dir / unique_filename

            content = await file.read()
            with open(file_path, 'wb') as f:
                f.write(content)

            upload_time = datetime.now()
            metadata_payload = {
                "document_id": document_id,
                "original_filename": original_name,
                "saved_filename": unique_filename,
                "upload_time": upload_time.isoformat(),
                "file_size": len(content),
            }
            metadata_path = _metadata_path_for(file_path)
            try:
                with open(metadata_path, "w", encoding="utf-8") as meta_file:
                    json.dump(metadata_payload, meta_file, ensure_ascii=False, indent=2)
            except OSError as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"写入元数据失败: {exc}",
                )

            queued_documents.append(
                QueuedDocument(
                    path=file_path,
                    document_id=document_id,
                    filename=original_name,
                    metadata={
                        "saved_filename": unique_filename,
                        "upload_time": upload_time.isoformat(),
                        "file_size": len(content),
                    },
                )
            )

            doc_info = DocumentInfo(
                document_id=document_id,
                filename=original_name,
                upload_time=upload_time,
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
        
        task_id = await task_manager.submit_task(queued_documents, options, tenant_id=tenant_id)
        
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
        knowledge_service = get_knowledge_service()
        data_dir = knowledge_service.data_dir
        
        if not data_dir.exists():
            return DocumentListResponse(
                documents=[],
                total_count=0,
                total_size=0
            )
        
        documents = []
        total_size = 0
        
        for file_path in data_dir.iterdir():
            if not file_path.is_file() or file_path.name.endswith(".meta.json"):
                continue

            file_stat = file_path.stat()
            metadata = _load_document_metadata(file_path)
            document_id = metadata.get("document_id") or f"doc_{file_path.stem}"
            display_name = metadata.get("original_filename") or file_path.name

            upload_time = datetime.fromtimestamp(file_stat.st_mtime)
            meta_upload_time = metadata.get("upload_time")
            if isinstance(meta_upload_time, str):
                try:
                    upload_time = datetime.fromisoformat(meta_upload_time)
                except ValueError:
                    pass

            file_size = metadata.get("file_size")
            if isinstance(file_size, (int, float)):
                file_size = int(file_size)
            else:
                file_size = file_stat.st_size

            total_size += file_size

            char_count = None
            if hasattr(knowledge_service, 'file_cache'):
                cache_key = str(file_path)
                cached_info = knowledge_service.file_cache.get(cache_key, {})
                char_count = cached_info.get('char_count')
            
            doc_info = DocumentInfo(
                document_id=document_id,
                filename=display_name,
                upload_time=upload_time,
                file_size=file_size,
                status=DocumentStatus.INDEXED,
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
        knowledge_service = get_knowledge_service()
        data_dir = knowledge_service.data_dir
        
        # 查找对应的文件
        deleted_files = []
        for file_path in data_dir.iterdir():
            if not file_path.is_file() or file_path.name.endswith(".meta.json"):
                continue

            metadata = _load_document_metadata(file_path)
            current_id = metadata.get("document_id")
            matches = current_id == document_id or (
                current_id is None and document_id in file_path.stem
            )
            if not matches:
                continue

            file_path.unlink()
            deleted_files.append(file_path.name)

            meta_path = _metadata_path_for(file_path)
            if meta_path.exists():
                try:
                    meta_path.unlink()
                except OSError:
                    pass
            
            if hasattr(knowledge_service, 'file_cache'):
                cache_key = str(file_path)
                knowledge_service.file_cache.pop(cache_key, None)
                knowledge_service._save_file_cache()
        
        if not deleted_files:
            raise HTTPException(
                status_code=404,
                detail=f"文档 {document_id} 不存在"
            )

        try:
            await knowledge_service.delete_document_async(document_id=document_id)
        except ValueError:
            pass
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"删除索引中的文档失败: {exc}"
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
async def rebuild_index(request: IndexRebuildRequest, tenant_id: str = Depends(get_tenant_id)):
    """
    重建向量索引
    
    - **parallel_workers**: 并行工作线程数
    - **force**: 是否强制重建
    """
    try:
        knowledge_service = get_knowledge_service()
        task_manager = get_task_manager()
        
        # 获取所有文档文件
        data_dir = knowledge_service.data_dir
        all_files = [f for f in data_dir.iterdir() if f.is_file()]
        
        if not all_files:
            raise HTTPException(
                status_code=400,
                detail="数据目录中没有文件，无法重建索引"
            )
        
        # 如果不是强制重建，检查索引是否已存在
        if not request.force and knowledge_service.has_index(tenant_id=tenant_id):
            return {
                "message": "索引已存在，使用 force=true 强制重建",
                "index_exists": True
            }
        
        # 异步重建索引
        documents = await knowledge_service.process_documents_async(all_files)
        await knowledge_service.build_index_async(documents, reset_existing=True, tenant_id=tenant_id)
        
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
        knowledge_service = get_knowledge_service()
        updated_config = {}
        
        # 更新最大并行工作线程数
        if config.max_parallel_workers is not None:
            knowledge_service.max_workers = config.max_parallel_workers
            updated_config['max_parallel_workers'] = config.max_parallel_workers
        
        # 更新embedding模型（需要重新初始化）
        if config.embedding_model is not None:
            knowledge_service._setup_embedding_model(config.embedding_model)
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
