import numpy as np
from collections import defaultdict
from faiss import IndexFlatIP, IndexFlatL2
from model2vec import StaticModel
from polars import DataFrame
from scipy.sparse import lil_matrix, dok_matrix
from sklearn.metrics.pairwise import euclidean_distances
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

# Exhaustively:
# 3 dimensional tensor of k document embedding matrices, D
# compute batch dot product of Eq and D
# max pool docs
# sum over Eq
#
# Pruning way:
# Maintain mapping of embedding to doc
# Faiss to get the unique docs
# Then go to exhaustive approach

doc_toks = lil_matrix((len(passages), len(potion.tokens)), dtype=np.int32)
for i, toks in enumerate(potion.tokenize(passages)):
    doc_toks[i, toks] = 1
doc_toks = doc_toks.tocsc()  # more efficient for wide sparse matrices

# human body blood
[1535, 1309, 1674]

# get the docs in columns matching the search tokens with np.nonzero
# and then get the tokens in those docs (likewise)


all_toks = np.ravel(I)
doc_idxs, _  = np.nonzero(doc_toks[:,all_toks])

results = []
for doc in all_docs:
    _, tok_idxs = np.nonzero(doc_toks[doc,:])
    doc_score = 0
    for qt in query_tokens:
        dists = ...
        doc_score += min(dists)
    results.append({ "pid": doc, "score": doc_score })
    break

query = "how many units of blood in a human body"

# Turns out in ColBERT they use negative squared L2 distance
# for the similarity metric, with norm'd vectors.
qembeds = potion.encode_as_sequence(query)
qembed_norm = np.linalg.norm(qembeds, axis=1, keepdims=True) + 1e-32

D, I = faiss_index.search(qembeds / qembed_norm, k=10)
D = -(D**2)
# D, I = nn.kneighbors(qembeds, n_neighbors=10, return_distance=True)
# D = 1 - D  # convert to sim


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
