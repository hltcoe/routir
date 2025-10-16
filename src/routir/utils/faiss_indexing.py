import argparse
from pathlib import Path

import faiss
import numpy as np
from tqdm.auto import tqdm


faiss.omp_set_num_threads(32)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        help="A directory with *.npy files, each containing a two varaibles: features and ids. "
        "`Ids` should be a list of document ids, and `features` should be a 2D matrix with "
        "the first dimenion matching the length of `ids`.",
    )
    parser.add_argument("output_dir", help="Output directory for the index")

    parser.add_argument("--index_string", default="PQ2048x4fs")
    parser.add_argument("--sampling_rate", type=float, default=0.07)

    parser.add_argument("--use_gpu", action="store_true", default=False)
    parser.add_argument("--two_step_training", action="store_true", default=False)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_fns = list(Path(args.input_dir).glob("*.npy"))

    sampled_fns = all_fns[:: int(1 / args.sampling_rate)]
    sampled_vectors = np.concatenate([np.load(fn, allow_pickle=True).item()["features"] for fn in tqdm(sampled_fns)], axis=0)

    # drop example with na
    sampled_vectors = sampled_vectors[~np.isnan(sampled_vectors).any(axis=1)]

    index = faiss.index_factory(sampled_vectors.shape[1], args.index_string, faiss.METRIC_INNER_PRODUCT)

    if args.use_gpu:
        if args.two_step_training:
            index_ivf = faiss.extract_index_ivf(index)
            clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatIP(index_ivf.d))
            index_ivf.clustering_index = clustering_index
        else:
            co = faiss.GpuMultipleClonerOptions()
            co.allowCpuCoarseQuantizer = True
            index = faiss.index_cpu_to_all_gpus(index)

    print("training...")
    index.train(sampled_vectors)

    if args.use_gpu:
        index = faiss.index_gpu_to_cpu(index)

    docids = []
    for fn in tqdm(all_fns, desc="adding", dynamic_ncols=True):
        shard = np.load(fn, allow_pickle=True).item()
        # dropna features
        mask = ~np.isnan(shard["features"]).any(axis=1)
        features = shard["features"][mask]
        ids = np.array(shard["ids"])[mask].tolist()

        index.add(features)
        docids += ids

    print("saving faiss index")
    faiss.write_index(index, str(output_dir / "index.faiss"))

    print("saving doc ids")
    with (output_dir / "index.ids").open("w") as fw:
        for docid in docids:
            fw.write(f"{docid}\n")
