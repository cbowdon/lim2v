def load_passages(*, limit: int | None):
    passage_ids = []
    passages = []
    with open("collections/msmarco-passage/collection.tsv") as f:
        for i, line in enumerate(f):
            if (
                limit is not None and i >= limit
            ):  # For demo/testing, use a subset. Remove for full run.
                break
            pid, text = line.strip().split("\t")
            passage_ids.append(pid)
            passages.append(text)
    return passage_ids, passages


def load_qrels(pids):
    qrels = {}
    pidset = set(pids)
    with open("collections/msmarco-passage/qrels.dev.small.tsv") as f:
        for line in f:
            qid, _, pid, relevance = line.strip().split("\t")
            if pid in pidset:
                qrels[qid] = {pid: int(relevance)}
    return qrels


def load_queries(qrels):
    queries = {}
    with open("collections/msmarco-passage/queries.dev.small.tsv") as f:
        for line in f:
            qid, query = line.strip().split("\t")
            if qid in qrels:
                queries[qid] = query
    return queries


def save_run(model_name: str, results: list[tuple[str, str, int, float]]):
    with open(f"results/{model_name}.txt", "w") as fout:
        for qid, pid, rank, score in results:
            fout.write(f"{qid} Q0 {pid} {rank} {score} {model_name}\n")


def eval_mrr(qrels, run):
    rrs = []
    for qid, qrel in qrels.items():
        if qid in run:
            for i, pid in enumerate(run[qid].keys()):
                if pid in qrel:
                    rank = i + 1
                    rrs.append(1.0 / rank)
                    break
            else:
                rrs.append(0.0)
    return sum(rrs) / len(rrs)
