import numpy as np
from collections import defaultdict
from faiss import IndexFlatIP, IndexFlatL2
from model2vec import StaticModel
from polars import DataFrame
from sklearn.neighbors import NearestNeighbors
from eval_sbert import load_passages, load_qrels, load_queries

potion = StaticModel.from_pretrained("minishlab/potion-base-8M", normalize=True)

l2_norm = np.linalg.norm(potion.embedding, axis=1, keepdims=True) + 1e-32
norm_embedding = potion.embedding / l2_norm

# faiss_index = IndexFlatIP(potion.dim)
# faiss_index.add(norm_embedding)

faiss_index = IndexFlatL2(potion.dim)
faiss_index.add(norm_embedding)

nn = NearestNeighbors(metric="cosine")
nn.fit(potion.embedding)

limit = 1000

pids, passages = load_passages(limit=limit)
df_p = DataFrame({"pid": pids, "passage": passages})

doc_index = defaultdict(set)
tids = potion.tokenize(passages)
for pid, _tids in zip(pids, tids):
    for tid in _tids:
        doc_index[tid].add(pid)

# for each query and document,
# for each query token q:
# what is the most sim document token in each document
# - and sum across query tokens
# if we get the top k vocab tokens for each q
# then we work down the ranks, assigning each doc its max score
# and sum across q

# Turns out in ColBERT they use negative squared L2 distance
# for the similarity metric, with norm'd vectors.
# Unfortunately this doesn't solve the problem anyway:
# the potion token weights are still not captured in distance so
# we lose the knowledge of what's a meaningful word

query = "how many units of blood in a human body"
qembeds = potion.encode_as_sequence(query)
qembed_norm = np.linalg.norm(qembeds, axis=1, keepdims=True) + 1e-32

D, I = faiss_index.search(qembeds / qembed_norm, k=10)
D = -np.pow(D, 2)
# D, I = nn.kneighbors(qembeds, n_neighbors=10, return_distance=True)
# D = 1 - D  # convert to sim

result_map = {}
for qt, ranked_vocab_tokens in enumerate(I):
    qt_map = {}
    for rank, tid in enumerate(ranked_vocab_tokens):
        docs = doc_index[tid]
        for doc in docs:
            if doc not in qt_map:
                qt_map[doc] = D[qt, rank].item()
    for doc, score in qt_map.items():
        if doc not in result_map:
            result_map[doc] = score
        else:
            result_map[doc] += score


d2 = (
    DataFrame(list(result_map.items()), schema=["pid", "score"], orient="row")
    .join(df_p, on="pid")
    .sort("score", descending=True)
)
d2.head()
