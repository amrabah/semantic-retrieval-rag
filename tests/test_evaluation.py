from app.evaluation import recall_at_k, reciprocal_rank, evaluate_rankings

def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"b", "x"}, 2) == 0.5

def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], {"c"}) == 1 / 3

def test_aggregate_metrics():
    out = evaluate_rankings([
        {"ranked_ids": ["a", "b"], "relevant_ids": ["a"]},
        {"ranked_ids": ["x", "c"], "relevant_ids": ["c"]},
    ], k=2)
    assert out["recall_at_k"] == 1.0
    assert out["mrr"] == 0.75
