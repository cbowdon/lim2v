# lim2v - can m2v address ColBERT's space usage without compromising accuracy?

This was an exploration of whether the "contextualised" part of contextualised
late-interaction BERT could be completely or partially replaced with weighted
static embeddings. This would ideally give us a model that exceeds vector search
with sentence embeddings for retrieval (as does ColBERT) but without the ColBERT
drawback of high space usage.

N.B. this is the end-to-end retrieval use case of ColBERT, not re-ranking.

## Outcomes

None of the models tried were able to outperform ColBERT on MSMARCO. There were
two major reasons:

1. MSMARCO passage with MRR@10 is a surprisingly problematic benchmark dataset
   for IR, despite its widespread usage. On reviewing the data it was easy to
   find examples of passages that were equivalent or superior to the "selected"
   passage for each query. As such MRR@10 is not reliable because it considers
   the rank of the single selected passage in each result set.

   At least one of the selected passages was also obviously false, which might
   not be directly problematic for the MRR evaluation but does raise questions
   in the context of retrieving accurate answers.

2. The use of static embeddings for the end-to-end retrieval is worse than BM25.
   The selected passage was in the initial rough result set only around 50% of
   the time, severely limiting the achievable score.

_So the reasons are the data, and the model. Sigh._

This prevented properly understanding how much of a benefit we see from the
contextualisation of embeddings. Clearly it matters, but I would still like to
know to what extent. Hopefully in future I'll have time to explore more.

## 0. Confirm BM25 baseline

This was very straightforward thanks to Pyserini's pre-compiled indices. I was
able to reproduce the 0.186 MRR@10 easily.

| Model           | Subset                       | MRR@10 |
| --------------- | ---------------------------- | ------ |
| BM25 (Pyserini) | 8.8M passages, 6,980 queries | 0.184  |

## 1. Eval sbert + faiss

Due to the time taken to vectorise all 8.8M passages in MSMARCO, I reduced down
to a 1M subset. However MRR@10 was poor, even after accounting for QIDs/PIDs not
in this subset. Upon manually judging a sample of results it became clear that
the MSMARCO passage annotations are inconsistent, with multiple equally correct
answers per query, and on occasion none of them selected as the actual answer.

Furthermore it turns out that the reciprocal rank calculation in `pytrec_eval`
has some surprising behaviour; it depends on the relevance score being
descending, even though relevance scores are not part of the reciprocal rank
calculation. Eventually I was able to configure the evaluation with
`pytrec_eval` so that it agreed with a manually calculated MRR, after which the
MRR@10 was higher as expected. My manually calculated MRR agreed with the
Anserini figures reported also.

| Model                    | Subset                   | MRR@10 |
| ------------------------ | ------------------------ | ------ |
| all-MiniLM-L6-v2 + FAISS | 1M passages, 238 queries | 0.32   |

## 2. Eval m2v + faiss

The naive run, dropping model2vec (potion) into the same experimental setup, did
not perform well; it was actually worse than the well-tuned BM25.

| Model                  | Subset                       | MRR@10 |
| ---------------------- | ---------------------------- | ------ |
| potion-base-8M + FAISS | 8.8M passages, 6,980 queries | 0.125  |

When I reviewed the ranked passages manually however, they were intuitive:
noticeably less accurate than sbert but good nonetheless. I think the problem
with MRR and MSMARCO is that the "selected" passages capture more than just
semantic relevance to the query, they capture the judge's feelings about how
authoritative and accurate the passage was. I will see if MaxSim helps at all,
but have lost faith in the dataset's suitability. (_Despite_ this being the
dataset and metric used by ColBERT.)

## 3. Implement and eval m2v + faiss + max-sim

There were some complexities to this. The sharing of (static) tokens across
documents means that there are too many doc matches for each token. It was
necessary to use potion's embedding weightings to re-score the results of the
initial token search and prune it quite aggressively.

| Model                            | Subset                    | MRR@10 |
| -------------------------------- | ------------------------- | ------ |
| potion-base-8M + FAISS + max sim | 100k passages, 34 queries | 0.20   |
| potion-base-8M + FAISS + max sim | 1M passages, 238 queries  | 0.11   |
| potion-base-8M + BM25 + max sim  | 8.8M passages, 50 queries | 0.14   |

_Note that the above aren't like-for-like, the subset size varies. Larger
subsets are harder so expect lower MRR@10. The reason for the variation is just
practical challenges in processing the data within the time available. I will
try and re-run the above with the full set of data but in the mean time you can
see what to expect from these interim results._

The approach has two main problems:

1. The initial rough search is too slow and has poor recall due to the large
   number of document hits. When using BM25 instead of dense embeddings for the
   initial search we see an improved MRR _despite_ running it on the full
   subset.
2. The "selected" passages of MSMARCO are frequently hard to justify over
   non-selected passages.

## Later:

- Contextual drift study (which embeddings move a lot? How are movements
  distributed across tokens?)
