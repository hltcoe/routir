# RoutIR: Fast Server for Hosting Retrieval Models for Retrieval-Augmented Generation

<div align="center">

[![PyPI version fury.io](https://badge.fury.io/py/routir.svg)](https://pypi.python.org/pypi/routir/)
[![Made with Python](https://img.shields.io/badge/Python->=3.9-blue?logo=python&logoColor=white)](https://python.org "Go to Python homepage")
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

</div>

RoutIR is a Python package that provides a simple and efficient wrapper around arbitrary retrieval models, including first stage retrieval, reranking, query expansion, and result fusion, and provides efficient asynchronous query batching and serving. 

## Get Started

You can install routir in your environment through `pip` or `uv`. 

```bash
pip install routir
```

RoutIR comes with a number of extras to install only the dependencies for the models you would like to serve. 
These extras include `dense`, `gpu`, `plaidx`, and `sparse`. 
You can instal any combinations, such as 

```bash
pip install "routir[dense,gpu]"
```

To start the service, simply provide the config file to the cli command `routir` to start. 
You can also optionally specify the port through `--port` flag (default 8000). 

```bash
routir config.json --port 5000
```

You can also use `uvx` to let `uv` creates a virtual environment on the fly for you:
```bash
uvx --with transformers --with torch routir config.json
```
Use `--with` to specify additional packages that you may need for serving the model. 
Please refer to `uv` [documentation](https://docs.astral.sh/uv/) for more information. 


## Configuration

The configuration file four major blocks: `services`, `collections`, `server_imports`, and `file_imports`.
- `services` and `collections` are list of object configuring each engine and each collection being served
- `server_imports` is list of external RoutIR endpoints that you would like to mirror in this endpoint. This will allow the end users to construct retrieval pipelines using services hosted on other machines. This is particularly helpful in a distributed compute cluster.
- `file_imports` is a list of custom Python scripts implemeting custom engines that RoutIR should load at initialization. More in the [Extension](#extension) section.


For example, if there are two other RoutIR instances running on `compute01:5000` and `compute02:5000` where each host `plaidx-neuclir` and `RankLllama`, you can import them as following. Users using this endpoint will be able to use both `plaidx-neuclir` and `RankLlama`. 
The following is an example config. 
```json
{
    "server_imports": [
        "http://compute01:5000",
        "http://compute02:5000",
    ],
    "file_imports": [
        "./examples/rank1_extension.py"
    ],
    "services": [
        {
            "name": "qwen3-neuclir",
            "engine": "Qwen3", 
            "cache": 1024, 
            "cache_ttl": 1024000, 
            "batch_size": 32, 
            "max_wait_time": 0.05, 
            "config": {
                "index_path": "hfds:routir/neuclir-qwen3-8b-faiss-PQ2048x4fs",
                "api_key": "YOUR_API_KEY_HERE OR AT OPENAI_API_KEY ENVIRONMENT VARIABLE",
                "embedding_base_url": "https://api.fireworks.ai/inference/v1/",
                "embedding_model_name": "accounts/fireworks/models/qwen3-embedding-8b",
                "k_scale": 5
            }
        },
        {
            "name": "rank1",
            "engine": "Rank1Engine",
            "config": {}
        }
    ],
    "collections": [
        {
            "name": "neuclir",
            "doc_path": "./neuclir-doc.jsonl"
        }
    ]
}
```

If you want to use Redis for caching, add `cache_redis_url` and `cache_redis_kwargs` to the `service` object. 
If your Redis instance is password-protected (which you should), add `password` field to `cache_redis_kwargs`. 

## HTTP API

1. Available services: GET `/avail`. An example output of the service initiated with the previous example config would be: 
```json
{
    "content": ["neuclir"],
    "score": ["Rank1", "RankLlama"],
    "search": ["qwen3-neuclir", "plaidx-neuclir"],
    "fuse": ["RRF", "ScoreFusion"], 
    "decompose_query": []
}
```

2. Search an index: POST `/search`. The following is an example request using `cURL`. 
```bash
curl -X POST http://localhost:5000/search \
-H "Content-Type: application/json" \
-d '{"service": "qwen3-neuclir", "query": "my test queries", "limit": 15}'
```
Output: 
```json
{
  "cached": true,
  "processed": true,
  "query": "my test queries",
  "scores": {
    "05a83946-dca2-4518-9bc3-3d394394d5e3": 0.3807981014251709,
    "36faf9fc-3751-4047-bb1c-2bd90fa6f4d4": 0.3675723671913147,
    "3a9ba832-f689-4204-8627-96abd73be65f": 0.42572247982025146,
    "6a5b81f3-9154-4959-9e88-79edfcecb43f": 0.3666379451751709,
    "6b086402-a00c-4fd8-8772-fade1f4b3198": 0.3996303975582123,
    "76ec4dd1-fb6e-4a1e-b3e2-4b6214886e52": 0.3723523020744324,
    "8c6e9e63-ea22-406e-a841-2dc645a3d2e2": 0.4014992415904999,
    "90f2e4af-8a92-4869-9c73-013fead4876d": 0.3644096851348877,
    "9dc749e8-f7a7-4c76-9883-03c7bc620d92": 0.37544310092926025,
    "aa3542e0-0c62-4518-9a0a-07eaa5b1eb00": 0.3768806755542755,
    "aeba1a4c-e02e-4d37-898c-68732c05b7d9": 0.3764134645462036,
    "b564d3aa-983d-42a4-b5ba-e6d43e79c094": 0.3760540783405304,
    "e46324a8-e9fb-442f-806d-1ed8f0efb2b0": 0.37497588992118835,
    "f91c5cf9-020b-4019-a483-41aee141808c": 0.3672129511833191,
    "fd6f8822-ddf4-4264-a449-5ecc7884c8ec": 0.36940526962280273
  },
  "service": "qwen3-neuclir",
  "timestamp": 1761023408.7890506
}
```

3. Score/Rerank a list of text given a query: POST `/score`. 
This allows you to score/rerank arbitrary pieces of text, such as document content, pasages in a document for context compression, 
or generated reponses for ranking answer relevancy. The following is an example request:
```bash
curl -X POST http://localhost:5000/score \
-H "Content-Type: application/json" \
-d '{
    "service": "rank1", 
    "query": "what is routir", 
    "passages": [
        "routir is a python package", 
        "sushi is the best food in the world"
    ]
}'
```
Output: 
```json
{
  "cached": false,
  "processed": true,
  "query": "what is routir",
  "scores": [
    0.9999997617631468,
    7.889264466868659e-06
  ],
  "service": "rank1",
  "timestamp": 1761026442.1780925
}

```

4. Search with dynamic pipeline: POST `/pipeline`. 
This allows the end users to construct an arbirary search pipeline with available engines on the fly. For example
```bash
curl -X POST http://localhost:5000/pipeline \
-H "Content-Type: application/json" \
-d '{
    "pipeline": "{qwen3-neuclir, plaidx-neuclir}RRF%50 >> rank1", 
    "query": "which team is the world series champion in 2020?",
    "collection": "neuclir"
}'
```
Output: 
```json
{
  "cached": false,
  "collection": "neuclir",
  "pipeline": "{qwen3-neuclir, plaidx-neuclir}RRF%50 >> rank1",
  "processed": true,
  "query": "which team is the world series champion in 2020?",
  "scores": {
    "027b3f6f-3dc6-4e69-86ae-2a98f8c4a881": 0.999999712631481,
    "066e645a-a495-4622-bcc8-7a804f598bcf": 5.4222202626709005e-06,
    "0ced1751-181a-4abb-8d64-37c362ede67c": 0.9999986290429566,
    "1c0d1e33-ea2c-48f3-9422-6f81259095eb": 0.9999996940976272,
    "27b429cc-b2a0-43cf-8b2b-883796486780": 4.539786865487149e-05,
    "2d11d0a3-78de-4201-ad26-64a6ac4b148f": 1.8925157266468097e-05,
    "302d1c1a-d620-4971-a44c-c1faead39494": 1.6701429809483402e-05,
    "39d2608d-e0a5-4b52-bb0e-b04968e21a15": 0.033085980653064666,
    "3c3c49f3-24b1-4dad-ac57-35b0565ab9b8": 2.1444943303118133e-05,
    "6e65cae3-443d-4cdd-9efc-8dfb3e1fe0b1": 6.962258739847376e-06,
    "7e4a4d57-9e73-4fb6-8ea7-584d0549c508": 0.0052201256185966365,
    "7ecdc77d-ea8c-4d48-9235-c21df9086831": 0.9999999397642365,
    "8660ca1b-ef5a-4c3a-a3e9-692e1e686f07": 1.9947301971022554e-06,
    "88b3eff5-738a-4bcd-b31a-15d9f1b9e198": 3.288748281343353e-06,
    "940bb6ff-f88a-40cd-88c1-2ae719d1dc74": 0.99999980249468,
    "a3edc861-7cf5-4152-a32b-90961bd12b80": 1.8925155010490798e-05,
    "c26f5a26-e732-4deb-80d6-b3ad6b249927": 0.9999986290426297,
    "da57d712-c6a8-4fa3-8e68-7f21ea7d3167": 0.9999996072138465,
    "ed231d01-05d9-4ed6-98d6-97b4e3e64aae": 0.00317268301626477,
    "f3954f32-62e6-4cb3-9ef2-78fe3dcb8f7a": 1.9947304348917116e-06
  },
  "service": "rank1",
  "timestamp": 1761026586.5823486
}
```




## Extension Examples

We provide several examples for integrating other IR toolkits with RoutIR. 
Please refer to each example for details. 

> [!WARNING]
> The Python script implementing the custom Engine needs to be imported through `file_imports` in the config. 
> When using `uvx`, remember to put the essential packages at `--with`. 

1. PyTerrier
```bash
python ./examples/pyterrier_extension.py # to build the index
uvx --with python-terrier routir ./examples/pyterrier_example_config.json --port 8000 # serve it at port 8000
```

2. Pyserini
```bash
uvx --with pyserini routir ./examples/pyserini_example_config.json --port 8000 # serve it at port 8000
```

3. Rank1
```bash
uvx --with mteb==1.39.0 --with vllm routir ./examples/rank1_example_config.json
```
The specific mteb version is crucial for this example. 


## Other Helper Scripts 
Here is an example command to generate `.npy` files containing Qwen3 document embeddings from a `.jsonl` file with `id`, `title`, and `text` fields:

```bash
python -m routir.utils.qwen3_encode /path/to/docs.jsonl /output/path \
--id-field id --fields title text --docs-per-file 10000
--batch-size 8 --model-name Qwen/Qwen3-Embedding-8B
```


To provide reference for the FAISS index structure that RoutIR uses, you can refer to the 
`routir.utils.faiss_indexing` for details. 
Here is an example command to generate a FAISS index from a directory containing `.npy` files, each with `features` and `ids` fields (as generated by the above script):

```bash
python -m routir.utils.faiss_indexing \
./encoded_vectors/ ./faiss_index.PQ2048x4fs.IP/ \
--index_string "PQ2048x4fs" --use_gpu --sampling_rate 0.25
```

## Contribution

We welcome any feedback, feature requests and pull requests. Please raise issues on GitHub.
Feel free to reach out to us through emails, ACM SIGIR Slack, or GitHub issues. 

## Attribution
TBA
