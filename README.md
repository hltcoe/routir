# routir: A Simple and Fast Search Service for Hosting State-of-the-Art Retrieval Models.

```bash
routir config.json
# or using uvx
uvx routir config
```

## Faiss Indexing 
```bash
python -m routir.utils.faiss_indexing \
./encoded_vectors/ ./faiss_index.PQ2048x4fs.IP/ \
--index_string "PQ2048x4fs" --use_gpu --sampling_rate 0.25
```

## Extension Examples

### PyTerrier
```bash
python ./examples/pyterrier_extension.py # to build the index
uvx --with python-terrier routir ./examples/pyterrier_example_config.json --port 8000 # serve it at port 8000
```

```python
import requests
requests.post("http://localhost:8000/search", json={"service": "pyterrier-cord", "query": "my test query", "limit": 15}).json()
```

### Pyserini
```bash
uvx --with pyserini routir ./examples/pyserini_example_config.json --port 8000 # serve it at port 8000
```

```python
import requests
requests.post("http://localhost:8000/search", json={"service": "pyserinibm25-neuclir-zho-dt", "query": "my test query", "limit": 15}).json()
```

