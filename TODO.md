## TODO

- [ ] Investigate replacing the `storage/` cache dependency now that vector indices live in Elasticsearch. We still rely on `storage/cache/file_cache.json` for deduping document ingestion; evaluate alternative cache locations or a toggle to disable the cache altogether.
