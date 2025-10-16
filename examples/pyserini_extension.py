from pathlib import Path

from pyserini.search.lucene import LuceneSearcher

from routir.models.abstract import Engine


class PyseriniBM25(Engine):
    def __init__(self, name: str = None, config=None, **kwargs):
        super().__init__(name, config, **kwargs)

        if self.index_path is not None and Path(self.index_path).exists():
            self.searcher = LuceneSearcher(self.index_path)
        else:
            assert "prebuilt_index_name" in self.config, (
                "No valid index_path provided, you should provide a `prebuilt_index_name`"
            )
            self.searcher = LuceneSearcher.from_prebuilt_index(self.config["prebuilt_index_name"])

        self.searcher.set_bm25(self.config.get("bm25_k1", 0.9), self.config.get("bm25_b", 0.4))
        self.searcher.set_rm3(
            self.config.get("rm3_fb_terms", 10),
            self.config.get("rm3_fb_docs", 10),
            self.config.get("rm3_original_query_weight", 0.5),
        )

    async def search_batch(self, queries, limit=20, **kwargs):
        if isinstance(limit, int):
            limit = [limit] * len(queries)
        assert len(limit) == len(queries)

        return [{docobj.docid: docobj.score for docobj in self.searcher.search(query, k=lm)} for query, lm in zip(queries, limit)]
