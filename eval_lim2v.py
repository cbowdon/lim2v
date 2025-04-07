import numpy as np
from collections import defaultdict
from faiss import IndexFlatIP
from model2vec import StaticModel
from eval_sbert import load_passages, load_qrels, load_queries

potion = StaticModel.from_pretrained("minishlab/potion-base-8M", normalize=True)

limit = 1000

pids, passages = load_passages(limit=limit)

faiss_index = IndexFlatIP(potion.dim)
faiss_index.train(potion.embedding)
faiss_index.add(potion.embedding)

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

query = potion.encode_as_sequence("how many units of blood in a human body")
D, I = faiss_index.search(query, k=10)

# TODO why are the distances not in [0, 1]?

result_map = defaultdict(int)
for qt, closest_to_query_token in enumerate(I):
    for rank, tid in enumerate(closest_to_query_token):
        docs = doc_index[tid]
        for doc in docs:
            result_map[doc] += 1 - D[qt, rank].item()
