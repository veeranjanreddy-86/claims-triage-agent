"""Tiny dependency-free BM25 index over the synthetic policy corpus.

Each markdown file has a small front-matter header (id, title, payer) followed by
``## Section`` blocks; every section is indexed as one chunk.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?")
_STOP = frozenset(
    (  # noqa: SIM905 - compact stop-word list
        "a an and are as at be by for from if in is it of on or the to with without this that "
        "should can may when must only not no"
    ).split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    title: str
    payer: str
    section: str
    text: str


def _parse_doc(raw: str) -> list[Chunk]:
    header, _, body = raw.partition("---\n")[2].partition("---\n")
    meta: dict[str, str] = {}
    for line in header.splitlines():
        key, sep, value = line.partition(":")
        if sep:
            meta[key.strip()] = value.strip().strip('"')
    chunks: list[Chunk] = []
    for block in re.split(r"^## ", body, flags=re.MULTILINE):
        if not block.strip():
            continue
        section, _, text = block.partition("\n")
        chunks.append(
            Chunk(
                meta["id"],
                meta["title"],
                meta.get("payer", "*"),
                section.strip(),
                " ".join(text.split()),
            )
        )
    return chunks


class PolicyCorpus:
    """BM25 (k1=1.5, b=0.75) over section chunks, with an optional payer filter."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        if not chunks:
            raise ValueError("policy corpus is empty")
        self.chunks = chunks
        self.k1, self.b = k1, b
        self._tfs = [Counter(tokenize(f"{c.title} {c.section} {c.text}")) for c in chunks]
        self._lens = [sum(tf.values()) for tf in self._tfs]
        self._avg_len = sum(self._lens) / len(self._lens)
        df: Counter[str] = Counter()
        for tf in self._tfs:
            df.update(tf.keys())
        n = len(chunks)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    @classmethod
    def from_directory(cls, directory: Path | None = None) -> PolicyCorpus:
        if directory is None:
            files = resources.files("claims_agent.data").joinpath("policies").iterdir()
            raws = [f.read_text(encoding="utf-8") for f in files if f.name.endswith(".md")]
        else:
            raws = [p.read_text(encoding="utf-8") for p in sorted(directory.glob("*.md"))]
        chunks = [c for raw in sorted(raws) for c in _parse_doc(raw)]
        return cls(chunks)

    def search(
        self, query: str, top_k: int = 3, payer: str | None = None
    ) -> list[tuple[Chunk, float]]:
        terms = tokenize(query)
        scored: list[tuple[Chunk, float]] = []
        for chunk, tf, length in zip(self.chunks, self._tfs, self._lens, strict=True):
            if payer and chunk.payer not in ("*", payer):
                continue
            score = 0.0
            for t in terms:
                f = tf.get(t, 0)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * length / self._avg_len)
                    score += self._idf[t] * f * (self.k1 + 1) / norm
            if score > 0:
                scored.append((chunk, round(score, 4)))
        scored.sort(key=lambda pair: (-pair[1], pair[0].doc_id, pair[0].section))
        return scored[:top_k]
