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
from collections import defaultdict
from faiss import IndexFlatL2
from model2vec import StaticModel
from numpy.typing import NDArray
from polars import DataFrame
from scipy.sparse import lil_matrix
from eval_sbert import load_passages, load_qrels, load_queries

potion = StaticModel.from_pretrained("minishlab/potion-base-8M", normalize=True)

l2_norm = np.linalg.norm(potion.embedding, axis=1, keepdims=True) + 1e-32
norm_tok_embeds = potion.embedding / l2_norm

faiss_index = IndexFlatL2(potion.dim)
faiss_index.add(norm_tok_embeds)

limit = 10_000

pids, passages = load_passages(limit=limit)
df_p = DataFrame({"pid": pids, "passage": passages})

WEIGHTS = np.abs(potion.embedding).sum(axis=1)
# tw = np.array(potion.tokens)[w.argsort()]
# df_tok_weights = DataFrame(dict(token=tw, weight=np.sort(w)))

doc_tok_mat = lil_matrix((len(passages), len(potion.tokens)), dtype=np.int32)
for i, toks in enumerate(potion.tokenize(passages)):
    doc_tok_mat[i, toks] = 1
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
    Eq: NDArray[np.float32], *, k: int = 10, weights: NDArray[np.float32] | None
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
    D_flat = D.reshape(-1, D.shape[2])
    S_flat = np.dot(D_flat, Eq.T)
    S = S_flat.reshape(D.shape[0], D.shape[1], Eq.shape[0])
    max_sims = S.max(axis=1)  # n_docs x n_query_toks
    scores = max_sims.sum(axis=1)  # n_docs
    sorting = scores.argsort()[::-1]
    return doc_ids[sorting], np.sort(scores)[::-1]


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


query = "how many units of blood in the human body"
Eq = embed(query)

doc_ids = rough_search(Eq, weights=query_weights(query))
doc_ids = np.arange(len(passages), dtype=np.int32)

results = exhaustive_search(Eq, doc_ids)

df_p[results[0]]


debug(Eq, passages[53])
# human body blood: [1535, 1309, 1674]


# We can use these weights to assist our rough search - and our max sim
# For the rough search, we could do some weighted scoring (e.g. wRRF) on the results of all query token searches
# which presumably will return primarily tokens matching the highest-weighted query token.
#
# In max-sim we could do a weighted sum.
