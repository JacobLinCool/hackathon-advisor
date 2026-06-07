from pathlib import Path


def test_main_interface_copy_is_builder_facing() -> None:
    html = Path("static/index.html").read_text(encoding="utf-8")
    app_js = Path("static/app.js").read_text(encoding="utf-8")
    combined = f"{html}\n{app_js}"

    assert "Under-explored directions" in html
    assert "Closest project echoes" in html
    assert "Press Plan to draft build steps for the selected idea." in app_js
    assert "Loading an example idea board." in app_js
    assert "Example idea board loaded with a plan and share page." in app_js
    assert "/api/artifact.png" in app_js
    assert "renderArtifactCanvas" not in app_js
    assert "canvas.toDataURL" not in app_js
    assert 'aria-label="Export build notes"' in html
    assert 'aria-label="Export idea-board chapter"' in html
    assert 'aria-label="Export current page as PNG"' in html
    assert "ideaCardAriaLabel" in app_js
    assert "Select idea:" in app_js

    stale_jargon = [
        "No wax path pressed.",
        "Gold has not gathered.",
        "No red ink yet.",
        "Demo rehearsal",
        "demo rehearsal",
        "press a new seal",
        "The page is choosing its words.",
        "Still riffling the inked pages.",
        "YOU VS THE WOOD",
        "current Wood",
        "advisor turns",
    ]
    for phrase in stale_jargon:
        assert phrase not in combined


def test_visible_static_shell_does_not_promote_submission_evidence() -> None:
    html = Path("static/index.html").read_text(encoding="utf-8").lower()

    promotional_terms = [
        "judge",
        "prize",
        "submission",
        "badge",
        "build-small",
        "hackathon criteria",
    ]
    for term in promotional_terms:
        assert term not in html


def test_server_png_copy_uses_interface_language() -> None:
    source = Path("hackathon_advisor/png_export.py").read_text(encoding="utf-8")

    assert "IDEA MAP" in source
    assert "YOU VS THE WOOD" not in source
