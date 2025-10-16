from pathlib import Path

import pandas as pd
import pyterrier as pt

from routir.models.abstract import Engine


class PyTerrier(Engine):
    def __init__(self, name: str = None, config=None, **kwargs):
        super().__init__(name, config, **kwargs)

        assert Path(self.index_path).exists()
        self.index = pt.IndexFactory.of(str(self.index_path.resolve()))

        wmodel = self.config.get("wmodel", "BM25")
        self.pipeline = (
            pt.terrier.Retriever(self.index, wmodel=wmodel)
            >> pt.rewrite.RM3(self.index)
            >> pt.terrier.Retriever(self.index, wmodel=wmodel)
        )

    async def search_batch(self, queries, limit=20, **kwargs):
        if isinstance(limit, int):
            limit = [limit] * len(queries)
        assert len(limit) == len(queries)

        rawdf: pd.DataFrame = self.pipeline(pd.Series(queries, name="query").rename_axis("qid").reset_index().astype("str"))
        return [
            d.sort_values("rank").iloc[:lm].set_index("docno")["score"].to_dict()
            for (qid, d), lm in zip(rawdf.groupby("qid", sort=True), limit)
        ]


if __name__ == "__main__":
    # short example to build the pyterrier index on TREC COVID
    ds = pt.datasets.get_dataset("irds:cord19/trec-covid")
    pt_index_path = "./examples/terrier_cord19"
    indexer = pt.index.IterDictIndexer(pt_index_path, text_attrs=["abstract"], meta={"docno": 20})
    index_ref = indexer.index(ds.get_corpus_iter())
