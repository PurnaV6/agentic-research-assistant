"""Formats the agent's final answer plus a de-duplicated source list into a report."""
from __future__ import annotations


class SourceStore:
    """Tracks every URL the agent actually looked at, in first-seen order.

    The final report appends this list independently of whatever citations
    the model remembered to write inline — the model can (and does)
    sometimes forget to cite a source it used, so the report's reference
    list is grounded in what tools were actually called, not in what the
    model claims it called.
    """

    def __init__(self):
        self._sources: dict[str, str] = {}  # url -> title

    def add(self, url: str, title: str = "") -> None:
        if not url:
            return
        if url not in self._sources or title:
            self._sources[url] = title or self._sources.get(url, "")

    def as_list(self) -> list[dict]:
        return [{"url": url, "title": title} for url, title in self._sources.items()]

    def __len__(self) -> int:
        return len(self._sources)


def build_report(answer: str, sources: list[dict]) -> str:
    lines = [answer.strip(), ""]
    if sources:
        lines.append("## Sources")
        for i, source in enumerate(sources, start=1):
            label = source["title"] or source["url"]
            lines.append(f"{i}. [{label}]({source['url']})")
    return "\n".join(lines)
