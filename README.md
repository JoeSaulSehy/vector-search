# Vector Search API

Flask API that serves FAISS + FinancialBERT search.

## Endpoints
- `GET /health` → "ok"
- `GET /search?q=QUERY&k=5&format=html|json`  
  Requires header: `X-API-Key: <SB_API_KEY>`

## How to build the index
Use your Colab notebook to chunk/embed and then:
```python
save_faiss(path_index="faiss_ip.index", path_meta="metadata.json")
