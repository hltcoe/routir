# routir: A Simple and Fast Search Service for Hosting State-of-the-Art Retrieval Models.

```bash
python -m routir.serve config.json
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
routir ./examples/pyterrier_example_config.json --port 8000 # serve it at port 8000
```

```python
import requests
requests.post("http://localhost:8000/search", json={"service": "pyterrier-cord", "query": "my test query", "limit": 15}).json()
```

### Pyserini
```bash
python ./examples/pyserini_extension.py # to build the index
routir ./examples/pyserini_example_config.json --port 8000 # serve it at port 8000
```

```python
import requests
requests.post("http://localhost:8000/search", json={"service": "pyserinibm25-neuclir-zho-dt", "query": "my test query", "limit": 15}).json()
```

