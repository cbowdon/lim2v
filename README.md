# lim2v - can m2v address ColBERT's space usage without compromising accuracy?

## 1. Confirm BM25 baseline

This was very straightforward thanks to Pyserini's pre-compiled indices. I was able to reproduce the 0.186 MRR@10 easily.

## 1. Eval sbert + faiss + cosine/inner product

Due to the time taken to vectorise all 8.8M passages in MSMARCO, I reduced down to a 1M subset. However MRR@10 was poor, even after accounting for QIDs/PIDs not in this subset. Upon manually judging a sample of results it became clear that the MSMARCO passage annotations are inconsistent, with multiple equally correct answers per query, and on occasion none of them selected as the actual answer.

Furthermore it turns out that the reciprocal rank calculation in `pytrec_eval` has some surprising behaviour; it depends on the relevance score being descending, even though relevance scores are not part of the reciprocal rank calculation. Eventually I was able to configure the evaluation with `pytrec_eval` so that it agreed with a manually calculated MRR, after which the MRR@10 was higher as expected. My manually calculated MRR agreed with the Anserini figures reported also.

## 2. Eval m2v + faiss + cosine
## 3. Implement m2v + faiss + max-sim

Later:

- Contextual drift study (which embeddings move a lot? How are movements distributed across tokens?)