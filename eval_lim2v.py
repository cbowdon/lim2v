#!/usr/bin/env python3

"""
ColBERT approach:

Exhaustively:
3 dimensional tensor of k document embedding matrices, D
compute batch dot product of Eq and D
max pool docs
sum over Eq

Pruning way:
Maintain mapping of embedding to doc
Faiss to get the unique docs
Then go to exhaustive approach
"""

import numpy as np
import pytrec_eval
import time
from collections import defaultdict
from faiss import IndexFlatL2
from itertools import batched
from model2vec import StaticModel
from numpy.typing import NDArray
from polars import DataFrame
from scipy.sparse import lil_matrix
from tqdm import tqdm
from lim2v.eval import *

print("Loading model")
potion = StaticModel.from_pretrained("minishlab/potion-base-8M", normalize=True)

l2_norm = np.linalg.norm(potion.embedding, axis=1, keepdims=True) + 1e-32
norm_tok_embeds = potion.embedding / l2_norm

WEIGHTS = np.abs(potion.embedding).sum(axis=1)
# tw = np.array(potion.tokens)[w.argsort()]
# df_tok_weights = DataFrame(dict(token=tw, weight=np.sort(w)))

print("Creating FAISS index")
faiss_index = IndexFlatL2(potion.dim)
faiss_index.add(norm_tok_embeds)

limit = 100_000

print("Loading passages")
pids, passages = load_passages(limit=limit)
# df_p = DataFrame({"pid": pids, "passage": passages})

print("Creating doc tok mat")  # takes about 20 mins for the whole thing
doc_tok_mat = lil_matrix((len(passages), len(potion.tokens)), dtype=np.int32)
batch_size = 64
for i, batch in enumerate(batched(tqdm(passages), n=batch_size)):
    for j, toks in enumerate(potion.tokenize(batch)):
        doc_tok_mat[i * batch_size + j, toks] = 1
doc_tok_mat = doc_tok_mat.tocsc()  # more efficient for wide sparse matrices


def embed(query: str) -> NDArray[np.float32]:
    qembeds = potion.encode_as_sequence(query)
    qembed_norm = np.linalg.norm(qembeds, axis=1, keepdims=True) + 1e-32
    return qembeds / qembed_norm


def query_weights(query: str) -> NDArray[np.float32]:
    toks = potion.tokenize([query])[0]
    return WEIGHTS[toks] / WEIGHTS.max()


def rrf(
    I: NDArray[np.int32], *, weights: NDArray[np.float32] | None = None
) -> NDArray[np.int32]:
    if weights is None:
        weights = np.ones(I.shape[0])

    # sum (w / k + r_i(d)) over d over i
    k = 60  # it just is
    doc_scores = defaultdict(int)
    for i, resultset in enumerate(I):
        for rank, doc in enumerate(resultset):
            doc_scores[doc] += weights[i] / (k + rank + 1)

    results = np.array(
        sorted(doc_scores.items(), key=lambda x: x[1], reverse=True),
        dtype=[("id", "<i4"), ("score", "<f4")],
    )
    return results["id"]


def rough_search(
    Eq: NDArray[np.float32], *, k: int = 5, weights: NDArray[np.float32] | None
) -> NDArray[np.int32]:
    D, I = faiss_index.search(Eq, k)
    ranked_toks = rrf(I, weights=weights)[:k]
    doc_idxs, _ = np.nonzero(doc_tok_mat[:, ranked_toks])
    result = np.unique(doc_idxs)
    return np.sort(result)


def pad_embeds(E: NDArray[np.float32], target: int) -> NDArray[np.float32]:
    n_to_pad = target - E.shape[0]
    if n_to_pad <= 0:
        return E
    pad_emb = norm_tok_embeds[0]
    pads = np.vstack([pad_emb for _ in range(n_to_pad)])
    return np.vstack((E, pads))


def get_doc_tok_mat(
    doc_ids: NDArray[np.int32], *, minlen: int = 0
) -> NDArray[np.float32]:
    _D = []
    maxlen = minlen
    for doc in doc_tok_mat[doc_ids]:
        _, toks = np.nonzero(doc)
        maxlen = max(maxlen, len(toks))
        _D.append(norm_tok_embeds[toks])

    pad_emb = norm_tok_embeds[0]
    D = [pad_embeds(_d, maxlen) for _d in _D]

    return np.array(D)


def exhaustive_search(Eq: NDArray[np.float32], doc_ids: NDArray[np.int32]):
    D = get_doc_tok_mat(doc_ids, minlen=Eq.shape[0])

    # Eq = torch.tensor(Eq, device="mps")
    # D = torch.tensor(D, device="mps")

    D_flat = D.reshape(-1, D.shape[2])
    S_flat = D_flat @ Eq.T
    S = S_flat.reshape(D.shape[0], D.shape[1], Eq.shape[0])
    max_sims = S.max(axis=1)  # n_docs x n_query_toks
    scores = max_sims.sum(axis=1)  # n_docs
    sorting = scores.argsort()[::-1]
    return np.array(
        list(zip(doc_ids[sorting], np.sort(scores)[::-1])),
        dtype=[("id", "<i4"), ("score", "<f4")],
    )


def debug(Eq: NDArray[np.float32], passage: str, k: int = 5):
    doc_toks = potion.tokenize([passage])[0]
    doc_toks_bow = np.unique(
        doc_toks
    )  # this uniqueness naturally enforced by doc_tok_mat
    norm_doc_embs = norm_tok_embeds[doc_toks_bow]
    S = np.dot(norm_doc_embs, Eq.T)
    max_sims = S.max(axis=1)
    top_toks = max_sims.argsort()[-k:][::-1]
    return [potion.tokens[doc_toks_bow[i]] for i in top_toks]


# query = "how long do you keep credit card statements"
# Eq = embed(query)
# weights = query_weights(query)

# doc_ids = rough_search(Eq, weights=weights)

# results = exhaustive_search(Eq, doc_ids)

# df_p[results["id"]]


# debug(Eq, passages[53])
# human body blood: [1535, 1309, 1674]

print("Loading queries")
qrels = load_qrels(pids)
queries = load_queries(qrels)

print("Running queries")
k = 10
results = []
for qid, query in tqdm(queries.items()):
    times = [("t0", time.perf_counter())]
    Eq = embed(query)
    times.append(("embed", time.perf_counter()))
    doc_ids = rough_search(Eq, weights=query_weights(query))
    times.append((f"rough {len(doc_ids):,}", time.perf_counter()))
    matches = exhaustive_search(Eq, doc_ids)[:k]
    times.append(("exhaustive", time.perf_counter()))
    for rank, (pid, score) in enumerate(matches):
        results.append((qid, pid.item(), rank + 1, score.item()))
    times = list(reversed(times))
    # print(qid, qrels[qid], matches["id"])
    # print([(name, t1 - t0) for (name, t1), (_, t0) in zip(times, times[1:])])

print("Saving and evaluating")
save_run("lim2v", results)

with open(f"results/lim2v.txt") as f:
    run = pytrec_eval.parse_run(f)

evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"recip_rank"})
results = evaluator.evaluate(run)

mrr = sum([metrics["recip_rank"] for metrics in results.values()]) / len(results)
print(f"MRR@10: {mrr:.4f}")

print(eval_mrr(qrels, run))
