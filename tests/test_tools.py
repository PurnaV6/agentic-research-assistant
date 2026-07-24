from unittest.mock import MagicMock, patch

from agent.tools import fetch_page, web_search


def test_web_search_returns_normalized_results():
    fake_results = [
        {"title": "Result A", "href": "https://a.example/1", "body": "snippet a"},
        {"title": "Result B", "href": "https://b.example/2", "body": "snippet b"},
    ]
    fake_ddgs = MagicMock()
    fake_ddgs.__enter__.return_value.text.return_value = fake_results
    fake_ddgs.__exit__.return_value = False

    with patch("agent.tools.DDGS", return_value=fake_ddgs):
        results = web_search("test query", max_results=2)

    assert results == [
        {"title": "Result A", "url": "https://a.example/1", "snippet": "snippet a"},
        {"title": "Result B", "url": "https://b.example/2", "snippet": "snippet b"},
    ]


def test_web_search_handles_exceptions_gracefully():
    with patch("agent.tools.DDGS", side_effect=RuntimeError("network down")):
        results = web_search("test query")

    assert len(results) == 1
    assert "error" in results[0]


def test_fetch_page_extracts_text():
    with patch("agent.tools.trafilatura.fetch_url", return_value="<html>...</html>"), patch(
        "agent.tools.trafilatura.extract", return_value="a" * 5000
    ):
        result = fetch_page("https://example.com/article", max_chars=100)

    assert result["url"] == "https://example.com/article"
    assert len(result["text"]) == 100


def test_fetch_page_reports_error_when_download_fails():
    with patch("agent.tools.trafilatura.fetch_url", return_value=None):
        result = fetch_page("https://example.com/missing")

    assert "error" in result


def test_fetch_page_reports_error_when_extraction_fails():
    with patch("agent.tools.trafilatura.fetch_url", return_value="<html></html>"), patch(
        "agent.tools.trafilatura.extract", return_value=None
    ):
        result = fetch_page("https://example.com/empty")

    assert "error" in result
