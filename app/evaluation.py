def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    retrieved = set(ranked_ids[:k])
    return len(retrieved & relevant_ids) / len(relevant_ids)

def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, item in enumerate(ranked_ids, start=1):
        if item in relevant_ids:
            return 1.0 / rank
    return 0.0

def evaluate_rankings(examples: list[dict], k: int = 5) -> dict[str, float]:
    if not examples:
        return {"recall_at_k": 0.0, "mrr": 0.0}
    recalls, rrs = [], []
    for ex in examples:
        relevant = set(ex["relevant_ids"])
        recalls.append(recall_at_k(ex["ranked_ids"], relevant, k))
        rrs.append(reciprocal_rank(ex["ranked_ids"], relevant))
    return {
        "recall_at_k": sum(recalls) / len(recalls),
        "mrr": sum(rrs) / len(rrs),
    }
