from hackathon_advisor.prize_ledger import prize_ledger


def test_prize_ledger_tracks_param_budget_and_badges() -> None:
    payload = prize_ledger({"backend": "rules", "model_id": "deterministic-tool-router"})

    assert payload["runtime"]["backend"] == "rules"
    assert payload["total_params_b"] <= payload["tiny_titan_limit_b"]
    assert payload["tiny_titan_eligible"] is True
    assert payload["largest_model"]["model"] == "openbmb/MiniCPM5-1B"
    badges = {badge["name"]: badge["status"] for badge in payload["badges"]}
    assert badges["Off the Grid"] == "ready"
    assert badges["Well-Tuned"] == "planned"
