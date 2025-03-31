import numpy as np
import pytrec_eval
from faiss import IndexFlatL2, IndexIVFPQ
from itertools import batched
from model2vec import StaticModel
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from typing import Literal

sbert = SentenceTransformer("all-MiniLM-L6-v2")
potion = StaticModel.from_pretrained("minishlab/potion-base-8M")


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


def sbert_embed(texts: list[str]):
    return sbert.encode(
        texts,
        output_value="sentence_embedding",
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def potion_embed(texts: list[str]):
    return potion.encode(
        texts,
        output_value="sentence_embedding",
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def build_ivfpq_index(embeds, *, dim: int):
    # using the params described in ColBERT
    nlist = 2000
    M = 16  # number of sub quantizers
    bits_per_vector = 8
    quantizer = IndexFlatL2(dim)
    index = IndexIVFPQ(quantizer, dim, nlist, M, bits_per_vector)
    index.train(embeds)
    index.add(embeds)
    index.nprobe = 10
    return index


def build_flat_index(embeds, *, dim: int):
    index = IndexFlatL2(dim)
    index.add(embeds)
    return index


def load_qrels(pids):
    qrels = {}
    pidset = set(pids)
    with open("collections/msmarco-passage/qrels.dev.small.tsv") as f:
        for line in f:
            qid, _, pid, relevance = line.strip().split("\t")
            if pid in pidset:
                qrels[qid] = {pid: int(relevance)}
    return qrels


def load_queries(qrels):
    queries = {}
    with open("collections/msmarco-passage/queries.dev.small.tsv") as f:
        for line in f:
            qid, query = line.strip().split("\t")
            if qid in qrels:
                queries[qid] = query
    return queries


def searcher(
    pids: list[int],
    passages: list[str],
    embed_fn=sbert_embed,
    index_fn=build_flat_index,
):
    embeds = embed_fn(passages)
    index = index_fn(embeds, dim=embeds.shape[1])

    def _search(queries: dict[str, str]):

        query_ids = list(queries.keys())
        query_texts = list(queries.values())
        query_embeddings = embed_fn(query_texts)

        D, I = index.search(query_embeddings, 10)  # top-10
        results = []
        for i, qid in enumerate(query_ids):
            for rank, pid_idx in enumerate(I[i]):
                pid = pids[pid_idx]
                score = 1 - D[i][rank]
                results.append((qid, pid, rank + 1, score.item()))
        return results

    return _search


def save_run(model_name: str, results: list[tuple[str, str, int, float]]):
    with open(f"results/{model_name}.txt", "w") as fout:
        for qid, pid, rank, score in results:
            fout.write(f"{qid} Q0 {pid} {rank} {score} {model_name}\n")


def eval_mrr(qrels, run):
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


limit = 100_000

pids, passages = load_passages(limit=limit)

qrels = load_qrels(pids)

queries = load_queries(qrels)

sbert_search = searcher(pids, passages, sbert_embed, build_flat_index)
potion_search = searcher(pids, passages, potion_embed, build_flat_index)

results = potion_search(queries)
results = sbert_search(queries)

save_run("dense-model", results)


# Load run file
with open(f"results/dense-model.txt") as f:
    run = pytrec_eval.parse_run(f)

# Evaluate
evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"recip_rank"})
results = evaluator.evaluate(run)

# Compute mean
mrr = sum([metrics["recip_rank"] for metrics in results.values()]) / len(results)
print(f"MRR@10: {mrr:.4f}")


print(eval_mrr(qrels, run))
