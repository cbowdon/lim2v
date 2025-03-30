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


def build_ivfpq_index(passages: list[str], *, dim: int):
    # using the params described in ColBERT
    nlist = 2000
    M = 16  # number of sub quantizers
    bits_per_vector = 8
    quantizer = IndexFlatL2(dim)
    index = IndexIVFPQ(quantizer, dim, nlist, M, bits_per_vector)
    embeds = embed_texts(passages)
    index.train(embeds)
    index.add(embeds)
    index.nprobe = 10
    return index


def build_flat_index(passages: list[str], *, dim: int):
    index = IndexFlatL2(dim)
    embeds = embed_texts(passages)
    index.add(embeds)
    return index


def load_qrels(pids):
    with open("collections/msmarco-passage/qrels.dev.small.tsv") as f:
        _qrels = pytrec_eval.parse_qrel(f)

    qrels = {}
    pidset = set(pids)
    for k, v in _qrels.items():
        if len(v) > 0:
            _pid = list(v.keys())[0]
            if _pid in pidset:
                qrels[k] = v
    return qrels


def load_queries(qrels):
    queries = {}
    with open("collections/msmarco-passage/queries.dev.small.tsv") as f:
        for line in f:
            qid, query = line.strip().split("\t")
            if qid in qrels:
                queries[qid] = query
    return queries


def searcher(pids: list[int], passages: list[str]):
    index = build_flat_index(passages, dim=384)

    def _search(query_ids: list[str], query_texts: list[str]):
        query_embeddings = embed_texts(query_texts)
        D, I = index.search(query_embeddings, 10)  # top-10
        results = []
        for i, qid in enumerate(query_ids):
            for rank, pid_idx in enumerate(I[i]):
                pid = pids[pid_idx]
                score = D[i][rank]
                results.append((qid, pid, rank + 1, score))
        return results

    return _search


limit = 100_000

pids, passages = load_passages(limit=limit)
index = build_flat_index(passages, dim=384)

qrels = load_qrels(pids)

queries = load_queries(qrels)

# Embed queries
query_ids = list(queries.keys())
query_texts = list(queries.values())
query_embeddings = embed_texts(query_texts)

# Search
D, I = index.search(query_embeddings, 10)  # top-10


with open("results/dense_run.txt", "w") as fout:
    for i, qid in enumerate(query_ids):
        for rank, pid_idx in enumerate(I[i]):
            pid = pids[pid_idx]
            score = 1 - D[i][rank]
            fout.write(f"{qid} Q0 {pid} {rank+1} {score} dense-model\n")


# Load run file
with open("results/dense_run.txt") as f:
    run = pytrec_eval.parse_run(f)

# Evaluate
evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"recip_rank"})
results = evaluator.evaluate(run)

# Compute mean
mrr = sum([metrics["recip_rank"] for metrics in results.values()]) / len(results)
print(f"MRR@10: {mrr:.4f}")

for i in range(5):
    _qr = qrels[query_ids[i]]
    print(query_ids[i], query_texts[i])
    for k, j in enumerate(I[i]):
        print(pids[j], k + 1, pids[j] in _qr, passages[j][:100], sep="\t")
    print("")


def mrr(qrels, run):
    rrs = []
    for qid, qrel in qrels.items():
        if qid in run:
            for i, pid in enumerate(run[qid].keys()):
                if pid in qrel:
                    rank = i + 1
                    rrs.append(1.0 / rank)
                    break
            else:
                rrs.append(0.0)
    return sum(rrs) / len(rrs)


mrr(qrels, run)
