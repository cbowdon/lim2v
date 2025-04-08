import numpy as np
from collections import defaultdict
from faiss import IndexFlatIP
from model2vec import StaticModel
from polars import DataFrame
from eval_sbert import load_passages, load_qrels, load_queries

potion = StaticModel.from_pretrained("minishlab/potion-base-8M", normalize=True)

l2_norm = np.linalg.norm(potion.embedding, axis=1, keepdims=True) + 1e-32
norm_embedding = potion.embedding / l2_norm

faiss_index = IndexFlatIP(potion.dim)
faiss_index.add(norm_embedding)

limit = 1000

pids, passages = load_passages(limit=limit)

doc_index = defaultdict(set)
tids = potion.tokenize(passages)
for pid, _tids in zip(pids, tids):
    for tid in _tids:
        doc_index[tid].add(pid)


# for each query and document,
# we want to get the max sim of every query token with every document token
# and sum these max sims
# so we first find the k document tokens that are closest to the query tokens
# make a map of document token to max query token similarity
# and then for each document with some of these tokens we sum the max sims

query = "how many units of blood in a human body"
qembeds = potion.encode_as_sequence(query)
qembeds /= np.linalg.norm(qembeds, axis=1, keepdims=True) + 1e-32

D, I = faiss_index.search(qembeds, k=10)

result_map = defaultdict(int)
for qt, closest_to_query_token in enumerate(I):
    for rank, tid in enumerate(closest_to_query_token):
        docs = doc_index[tid]
        for doc in docs:
            result_map[doc] += D[qt, rank].item()

d2 = DataFrame(list(result_map.items()), schema=["doc_id", "score"], orient="row").sort(
    "score", descending=True
)
d2.head()
