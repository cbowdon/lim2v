import pytrec_eval
from pyserini.search import get_qrels_file, get_topics
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm

# Load prebuilt BM25 index
searcher = LuceneSearcher.from_prebuilt_index('msmarco-passage')
searcher.search("What's the time?")
topics = get_topics("msmarco-passage-dev-subset")

# Store results in TREC format
with open('bm25_run.txt', 'w') as fout:
    for qid, query in topics.items():
        hits = searcher.search(query["title"], k=10)
        for rank, hit in enumerate(hits):
            fout.write(f"{qid} Q0 {hit.docid} {rank+1} {hit.score} bm25\n")




# Load qrels
qrel_file = get_qrels_file('msmarco-passage-dev-subset')
with open(qrel_file) as f:
    qrels = pytrec_eval.parse_qrel(f)

# Load run file
with open('bm25_run.txt') as f:
    run = pytrec_eval.parse_run(f)

# Evaluate
evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'recip_rank'})
results = evaluator.evaluate(run)

# Compute mean
mrr = sum([metrics['recip_rank'] for metrics in results.values()]) / len(results)
print(f"MRR@10: {mrr:.4f}")