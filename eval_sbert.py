import numpy as np
import pytrec_eval
from faiss import IndexFlatL2, IndexIVFPQ
from itertools import batched
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from typing import Literal

biencoder = SentenceTransformer("all-MiniLM-L6-v2")


def load_passages(*, limit: int | None):
    passage_ids = []
    passages = []
    with open("collections/msmarco-passage/collection.tsv") as f:
        for i, line in enumerate(f):
            if (
                limit is not None and i >= limit
            ):  # For demo/testing, use a subset. Remove for full run.
                break
            pid, text = line.strip().split("\t")
            passage_ids.append(pid)
            passages.append(text)
    return passage_ids, passages


def embed_texts(texts: list[str]):
    return biencoder.encode(
        texts,
        output_value="sentence_embedding",
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def build_index(*, dim: int, limit: int | None):

    # using the params described in ColBERT
    nlist = 2000
    M = 16  # number of sub quantizers
    bits_per_vector = 8
    quantizer = IndexFlatL2(dim)
    index = IndexIVFPQ(quantizer, dim, nlist, M, bits_per_vector)

    pids, passages = load_passages(limit=limit)
    embeds = embed_texts(passages)

    index.train(embeds)
    index.add(embeds)
    index.nprobe = 10
    return pids, passages, index


pids, passages, index = build_index(dim=384, limit=1_000_000)

# queries = embed_texts(["what is skimmed milk?"])
# index.search(queries, k=10)


queries = {}
with open("collections/msmarco-passage/queries.dev.small.tsv") as f:
    for line in f:
        qid, query = line.strip().split("\t")
        queries[qid] = query

# Embed queries
query_ids = list(queries.keys())
query_texts = list(queries.values())
query_embeddings = embed_texts(query_texts)

# Search
D, I = index.search(query_embeddings, 10)  # top-10


with open("dense_run.txt", "w") as fout:
    for i, qid in enumerate(query_ids):
        for rank, pid_idx in enumerate(I[i]):
            pid = pids[pid_idx]
            score = D[i][rank]
            fout.write(f"{qid} Q0 {pid} {rank+1} {score} dense-model\n")


# Load qrels
with open("collections/msmarco-passage/qrels.dev.small.tsv") as f:
    _qrels = pytrec_eval.parse_qrel(f)

qrels = {}
pidset = set(pids)
for k, v in _qrels.items():
    if len(v) > 0:
        _pid = list(v.keys())[0]
        if _pid in pidset:
            qrels[k] = v

# Load run file
with open("dense_run.txt") as f:
    run = pytrec_eval.parse_run(f)

# Evaluate
evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"recip_rank"})
results = evaluator.evaluate(run)

# Compute mean
mrr = sum([metrics["recip_rank"] for metrics in results.values()]) / len(results)
print(f"MRR@10: {mrr:.4f}")
