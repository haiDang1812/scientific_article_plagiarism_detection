# Scientific Article Plagiarism Detection

## Embedding Server Storage Modes

`embed_server.py` now supports 2 ChromaDB storage modes:

- `local` (default): persist vectors to local disk.
- `cloud`: store vectors on a remote ChromaDB server/cloud endpoint.

## Local Mode (default)

Run with local persistent storage:

```powershell
python embed_server.py --chroma-mode local --chroma-path ./.chroma_db
```

Or use environment variables:

```powershell
$env:CHROMA_MODE="local"
$env:CHROMA_LOCAL_PATH="./.chroma_db"
python embed_server.py
```

## Cloud Mode (ChromaDB)

Set cloud connection params with CLI:

```powershell
python embed_server.py `
	--chroma-mode cloud `
	--chroma-cloud-tenant "your_tenant" `
	--chroma-cloud-database "your_database" `
	--chroma-cloud-api-key "your_api_key"
```

Or with environment variables:

```powershell
$env:CHROMA_MODE="cloud"
$env:CHROMA_CLOUD_TENANT="your_tenant"
$env:CHROMA_CLOUD_DATABASE="your_database"
$env:CHROMA_CLOUD_API_KEY="your_api_key"
python embed_server.py
```

Notes:

- Keep `--chroma-cloud-api-key` (or `CHROMA_CLOUD_API_KEY`) secret.
- If your Chroma server does not use tenant/database routing, the server falls back to a compatible HTTP client mode automatically.

## Run command
uv run uvicorn embed_server.server:app --host 0.0.0.0 --port 8000