from agent.report import SourceStore, build_report


def test_source_store_dedupes_by_url_and_keeps_first_seen_order():
    store = SourceStore()
    store.add("https://a.example", "A")
    store.add("https://b.example", "B")
    store.add("https://a.example", "")  # re-add without title shouldn't overwrite title

    assert len(store) == 2
    assert store.as_list() == [
        {"url": "https://a.example", "title": "A"},
        {"url": "https://b.example", "title": "B"},
    ]


def test_source_store_ignores_empty_url():
    store = SourceStore()
    store.add("", "no url")
    assert len(store) == 0


def test_build_report_includes_numbered_sources():
    report = build_report("The answer is 42.", [{"url": "https://a.example", "title": "A"}])
    assert "The answer is 42." in report
    assert "## Sources" in report
    assert "1. [A](https://a.example)" in report


def test_build_report_omits_sources_section_when_empty():
    report = build_report("No evidence needed.", [])
    assert "## Sources" not in report
