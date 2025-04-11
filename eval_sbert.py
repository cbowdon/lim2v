#!/usr/bin/env python3

import numpy as np
import pytrec_eval
from collections import defaultdict
from faiss import IndexFlatL2, IndexIVFPQ
from itertools import batched
from model2vec import StaticModel
from numpy.typing import NDArray
from pprint import pprint
from sentence_transformers import SentenceTransformer
from typing import Literal
from lim2v.eval import *

sbert = SentenceTransformer("all-MiniLM-L6-v2")
potion = StaticModel.from_pretrained("minishlab/potion-base-8M", normalize=True)


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
    return potion.encode(texts, show_progress_bar=True)


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


def searcher(
    pids: list[int],
    passages: list[str],
    embed_fn=sbert_embed,
    index_fn=build_flat_index,
):
    embeds = embed_fn(passages)
    index = index_fn(embeds, dim=embeds.shape[1])

    def _search(queries: dict[str, str], k: int = 10, return_passages: bool = False):

        query_ids = list(queries.keys())
        query_texts = list(queries.values())
        query_embeddings = embed_fn(query_texts)

        D, I = index.search(query_embeddings, k)

        results = defaultdict(dict)
        if return_passages:
            for i, qid in enumerate(query_ids):
                for rank, pid_idx in enumerate(I[i]):
                    pid = pids[pid_idx]
                    passage = passages[pid_idx]
                    results[queries[qid]][pid] = passage
            return results

        results = []
        for i, qid in enumerate(query_ids):
            for rank, pid_idx in enumerate(I[i]):
                pid = pids[pid_idx]
                score = D[i][rank]
                results.append((qid, pid, rank + 1, score.item()))
        return results

    return _search


if __name__ == "__main__":
    limit = 1_000_000

    pids, passages = load_passages(limit=limit)

    qrels = load_qrels(pids)

    queries = load_queries(qrels)

    print(f"{len(passages):,} passages, {len(queries):,} queries")

    # sbert_search = searcher(pids, passages, sbert_embed, build_flat_index)
    potion_search = searcher(pids, passages, potion_embed, build_flat_index)

    # results = sbert_search(queries)
    results = potion_search(queries, return_passages=False)
    # for k, v in results.items():
    #    print(f"# {k}")
    #    for k_, v_ in v.items():
    #        print(f"-- {k_}: {v_}")
    # assert False  # exit early after debugging

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
