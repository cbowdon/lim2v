"""
The aim here is to analyse how tokens move when contextualised. Do some tokens move a lot and some a little?
Intuitively, some are more context-dependent than others.
"""
import torch
from polars import DataFrame, col
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from lim2v.eval import load_passages

sbert = SentenceTransformer("all-MiniLM-L6-v2")

pids, texts = load_passages(limit=100)


#def sbert_embed(texts: list[str]):

tokens = sbert.tokenize(texts)["input_ids"]

embeds = sbert.encode(
    texts,
    output_value="token_embeddings",
    batch_size=32,
    convert_to_tensor=True,
    normalize_embeddings=True,
    show_progress_bar=True,
)


tidtoks = {tid: tok for tok, tid in sbert.tokenizer.vocab.items()}
tokmap = {tid: [] for tid in range(len(sbert.tokenizer.vocab))}
for i in tqdm(range(len(texts))):
    toks = tokens[i]
    embs = embeds[i]
    for tok, emb in zip(toks, embs):
        if tok == 0:
            break
        tokmap[tok.item()].append(emb)


tokdists = []
for tid, v in tqdm(tokmap.items()):
    if len(v) == 0:
        continue  # token never appeared
    contextual_embs = torch.vstack(v)
    dists = torch.pairwise_distance(contextual_embs, contextual_embs)
    for d in dists:
        tokdists.append({"tid": tid, "dist": d.item()})

df = DataFrame(tokdists)

# So now we have the distances between different contextualised versions of tokens
# Let's tidy the data and ask questions like mean movement and max movement
