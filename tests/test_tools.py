from hackathon_advisor.tools import idea_from_text


def test_idea_from_text_splits_explicit_title_and_pitch() -> None:
    title, pitch = idea_from_text(
        "idea: Hands-on science coach -- A lab-notebook companion for household experiments."
    )

    assert title == "Hands-on science coach"
    assert pitch == "A lab-notebook companion for household experiments."
