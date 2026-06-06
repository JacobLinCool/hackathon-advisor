from hackathon_advisor.aliases import normalize_text


def test_normalize_jargon_alias() -> None:
    text, corrections = normalize_text("use neutron with mini cpm on zero gpu")

    assert "Nemotron" in text
    assert "MiniCPM5" in text
    assert "ZeroGPU" in text
    assert {item.canonical for item in corrections} >= {"Nemotron", "MiniCPM5", "ZeroGPU"}
