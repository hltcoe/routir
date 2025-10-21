#!/usr/bin/env python3
"""
Pipeline Evaluator

This script queries and evaluates a routir pipeline using queries and qrels from an ir_dataset.

Usage:
    python evaluate_pipeline.py <routir endpoint> [options]

Example:
    python evaluate_pipeline.py localhost:5000 --pipeline plaidx-neuclir --limit 100 --queries neuclir/1/multi/trec-2023 --qrels neuclir/1/multi/trec-2023
"""

import argparse
import asyncio

import aiohttp
import ir_datasets as irds
import ir_measures as irms
from tqdm.asyncio import tqdm_asyncio


async def async_avail(endpoint):
    timeout = aiohttp.ClientTimeout(total=6000)
    async with aiohttp.ClientSession() as session:
        async with session.get(endpoint + "/avail", timeout=timeout) as response:
            return await response.json()


async def async_search(endpoint, pipeline_name, query, collection, semaphore, limit=5, **kwargs):
    data = {"pipeline": pipeline_name, "query": str(query), "limit": limit, "collection": collection, **kwargs}
    async with semaphore:
        timeout = aiohttp.ClientTimeout(total=6000)
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint + "/pipeline", json=data, timeout=timeout) as response:
                return await response.json()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("endpoint", type=str, help="routir endpoint")
    parser.add_argument("--pipeline", type=str, default="plaidx-neuclir", help="Pipeline to query (default: %(default)s)")
    parser.add_argument("--collection", type=str, default=None, help="Collection (default: %(default)s)")
    parser.add_argument("--limit", type=int, default=100, help="Number of documents to return (default: %(default)s)")
    parser.add_argument("--concurrency", type=int, default=50, help="Maximum concurrent queries (default: %(default)s)")
    parser.add_argument(
        "--queries", type=str, default="neuclir/1/multi/trec-2023", help="ir-dataset providing queries (default: %(default)s)"
    )
    parser.add_argument(
        "--qrels", type=str, default="neuclir/1/multi/trec-2023", help="ir-dataset providing qrels (default: %(default)s)"
    )

    args = parser.parse_args()

    if not args.endpoint.startswith("http"):
        args.endpoint = "http://" + args.endpoint

    if args.collection.lower() == "none":
        args.collection = None

    avail = await async_avail(args.endpoint)
    if args.collection and args.collection not in avail["content"]:
        print(f"collection {args.collection} is not present in the endpoint content: {avail['content']}")
        raise ValueError(f"unknown collection: {args.collection}")

    semaphore = asyncio.Semaphore(args.concurrency)

    qids, queries = zip(
        *[(query.query_id, " ".join((query.title, query.description))) for query in irds.load(args.queries).queries_iter()]
    )

    results = await tqdm_asyncio.gather(
        *[async_search(args.endpoint, args.pipeline, query, args.collection, semaphore, limit=args.limit) for query in queries],
        desc=args.queries,
    )

    print(
        irms.calc_aggregate(
            [irms.nDCG @ 20],
            irds.load(args.qrels).qrels_iter(),
            {qid: r["scores"] for qid, r in zip(qids, results)},
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
