## TODO

- [ ] Investigate replacing the `storage/` cache dependency now that vector indices live in Elasticsearch. We still rely on `storage/cache/file_cache.json` for deduping document ingestion; evaluate alternative cache locations or a toggle to disable the cache altogether.
- [ ] 统一服务日志配置，整理 logger 级别与 handlers，避免重复输出并支持环境变量控制。
