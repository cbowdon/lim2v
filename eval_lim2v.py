import numpy as np
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

doc_tok_mat = lil_matrix((len(passages), len(potion.tokens)), dtype=np.int32)
for i, toks in enumerate(potion.tokenize(passages)):
    doc_tok_mat[i, toks] = 1
doc_tok_mat = doc_tok_mat.tocsc()  # more efficient for wide sparse matrices


def embed(query: str) -> NDArray[np.float32]:
    qembeds = potion.encode_as_sequence(query)
    qembed_norm = np.linalg.norm(qembeds, axis=1, keepdims=True) + 1e-32
    return qembeds / qembed_norm


def rough_search(Eq: NDArray[np.float32], k: int = 5) -> NDArray[np.int32]:
    """With non-contextualised embeddings, this is less useful because of high overlap - needs weighting/pruning."""
    D, I = faiss_index.search(Eq, k)
    all_toks = np.unique(np.ravel(I))
    doc_idxs, _ = np.nonzero(doc_tok_mat[:, all_toks])
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
    doc_toks_bow = np.unique(doc_toks)
    norm_doc_embs = norm_tok_embeds[doc_toks_bow]
    S = np.dot(norm_doc_embs, Eq.T)
    max_sims = S.max(axis=1)
    top_toks = max_sims.argsort()[-k:][::-1]
    return [potion.tokens[doc_toks_bow[i]] for i in top_toks]


Eq = embed("how many units of blood in the human body")

doc_ids = rough_search(Eq)
doc_ids = np.arange(len(passages), dtype=np.int32)

results = exhaustive_search(Eq, doc_ids)

debug(Eq, passages[53])

df_p[results[0]]

# human body blood: [1535, 1309, 1674]
