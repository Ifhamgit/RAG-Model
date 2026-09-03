"""Chunker — split on meaning boundaries the author already wrote (DESIGN.md §b).

Two ideas do most of the work here.

**Structure first.** These documents are heavily marked up by their authors —
banner sections, numbered clauses, `Q3.6`-style FAQ pairs, program headings. A
blind character window would cut §2.2's worked refund example in half, which is
exactly the content an agent needs whole. Size-based splitting is the fallback
for an oversized structural unit, not the primary mechanism.

**Breadcrumb headers.** Every chunk is prefixed with `[title | doc_id | §section]`
*inside the indexed text*. "70% refund, GST fully refunded" is nearly
meaningless alone; with its breadcrumb it is retrievable, and a query mentioning
"refund policy" matches a chunk whose body never says the word "policy". This is
the single highest-leverage line in the module (§b.2).
"""

from __future__ import annotations

import hashlib
import re
import statistics
from typing import Iterable, Iterator, Optional

from .config import Settings
from .loaders import DOC_ID_RE
from .models import Authority, Chunk, RawDoc

# --------------------------------------------------------------------------
# structural markers
# --------------------------------------------------------------------------

# A banner rule: the "====" / "----" lines that fence a section title in the
# plain-text policy documents.
RULE_RE = re.compile(r"^\s*([=\-_])\1{4,}\s*$")

# A numbered clause opening a new unit: "2.1 The refund payable ...", "5.3 ...".
CLAUSE_RE = re.compile(r"^\s*(\d+\.\d+)\s+\S")

# An FAQ question: "Q3.6 Do you offer an Income Share Agreement?"
FAQ_Q_RE = re.compile(r"^\s*Q\d+\.\d+\s")

# PDF headings are detected BY SHAPE. The text pypdf extracts carries no markup
# — the "##" markers existed only in the script that generated the PDF and are
# not in the file. Each entry is one recognisable heading shape; a missed
# heading is a one-line addition here, which is why they are kept in one list.
PDF_HEADING_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"^[A-Z]{2,4}-\d{3}\b"),              # SEF-101 — Software Engineering...
    re.compile(r"^Module\s+\d+"),                    # Module 4 — Databases and SQL
    re.compile(r"^\d+(\.\d+)*\.?\s+[A-Za-z]"),       # 5.3 Learner conditions
    re.compile(r"^Section\s+\d+", re.IGNORECASE),
)

# Headings that are just a noun phrase, with no structural marker to key off.
PDF_HEADING_TITLES: frozenset[str] = frozenset(
    {
        "syllabus", "capstone projects", "career outcomes", "instructors",
        "certification", "next steps", "about meridian academy",
        "how our programs are structured", "reported outcomes",
        "hiring partner network", "purpose and scope",
    }
)

# A run of 3+ spaces is the signature of a rendered table row. Without this
# guard the placement policy's outcomes table matches the program-code heading
# regex — "SEF-101    Yes    No" would be promoted to a section title.
TABLE_ROW_RE = re.compile(r"\S {3,}\S")

MAX_HEADING_CHARS = 80
# 14, not 10: "Module 5 — Natural Language Processing and Large Language Models
# (8 weeks)" is a genuine heading at 12 words. The character limit is the real
# guard against prose; the word count only catches long thin lines.
MAX_HEADING_WORDS = 14

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
_PARA_SPLIT = re.compile(r"\n\s*\n")


# --------------------------------------------------------------------------
# a structural unit, before sizing
# --------------------------------------------------------------------------


class _Unit:
    """A candidate chunk: some text plus the section it belongs to."""

    __slots__ = ("text", "section", "page", "atomic")

    def __init__(self, text: str, section: str, page: Optional[int], atomic: bool = False):
        self.text = text.strip()
        self.section = section
        self.page = page
        # `atomic` units are never merged with a neighbour and never dropped for
        # being short. FAQ pairs are atomic: the question line is the retrieval
        # signal, and gluing two questions together blurs both (§b.3).
        self.atomic = atomic

    def __len__(self) -> int:
        return len(self.text)


# --------------------------------------------------------------------------
# per-type structural splitters
# --------------------------------------------------------------------------


def _iter_banner_sections(text: str) -> Iterator[tuple[str, list[str]]]:
    """Yield (section_title, body_lines) for banner-fenced text documents.

    Handles both the "====" top-level banners and the "----" sub-banners the
    eligibility document uses for its per-program blocks (3.1 to 3.5). The rule
    characters are stripped from the body; the title survives as metadata and,
    via the breadcrumb, as indexed text.
    """
    lines = text.split("\n")
    section = ""
    buf: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # A title is a line fenced by rule lines above and below.
        # ...but not a rule-fenced *table header*. The refund schedule in §2.1 is
        # drawn as "----- / Refund Day Count  Tuition refunded  Deductions / -----",
        # which is structurally identical to a section banner. Promoting it to a
        # section title puts a column header into every breadcrumb in the most
        # important table in the corpus, so table-shaped candidates are rejected.
        if (
            RULE_RE.match(line)
            and i + 2 < len(lines)
            and lines[i + 1].strip()
            and RULE_RE.match(lines[i + 2])
            and not TABLE_ROW_RE.search(lines[i + 1])
        ):
            if buf:
                yield section, buf
                buf = []
            section = lines[i + 1].strip()
            i += 3
            continue
        if RULE_RE.match(line):  # an unpaired rule; drop the decoration
            i += 1
            continue
        buf.append(line)
        i += 1
    if buf:
        yield section, buf


def _split_text(doc: RawDoc) -> list[_Unit]:
    """Banner sections, then numbered clauses within each (§b.4)."""
    units: list[_Unit] = []
    for section, body_lines in _iter_banner_sections(doc.text):
        current: list[str] = []
        for line in body_lines:
            # A numbered clause starts a new unit. The number stays in the body
            # because agents cite "§5.3" and it must be retrievable.
            if CLAUSE_RE.match(line) and current and "\n".join(current).strip():
                units.append(_Unit("\n".join(current), section, doc.page))
                current = [line]
            else:
                current.append(line)
        if "\n".join(current).strip():
            units.append(_Unit("\n".join(current), section, doc.page))
    return units


def _split_faq(doc: RawDoc) -> list[_Unit]:
    """One atomic unit per Q&A pair (§b.4).

    The clearest case for structure-aware chunking in this corpus: a Q&A pair is
    already a self-contained retrieval unit, and its question line is a natural
    paraphrase of what a user will type. A question is never separated from its
    answer, even when the pair exceeds MAX_CHARS.
    """
    units: list[_Unit] = []
    for section, body_lines in _iter_banner_sections(doc.text):
        current: list[str] = []
        for line in body_lines:
            if FAQ_Q_RE.match(line):
                if "\n".join(current).strip():
                    units.append(_Unit("\n".join(current), section, doc.page, atomic=True))
                current = [line]
            else:
                current.append(line)
        if "\n".join(current).strip():
            units.append(_Unit("\n".join(current), section, doc.page, atomic=True))
    return units


def is_pdf_heading(line: str) -> bool:
    """Shape-based heading detection for extracted PDF text."""
    s = line.strip()
    if not (3 <= len(s) <= MAX_HEADING_CHARS):
        return False
    if s[-1] in ".,;:":
        return False
    if RULE_RE.match(s) or TABLE_ROW_RE.search(s):
        return False
    if len(s.split()) > MAX_HEADING_WORDS:
        return False
    if s.lower() in PDF_HEADING_TITLES:
        return True
    return any(rx.match(s) for rx in PDF_HEADING_RES)


def _split_pdf(doc: RawDoc, carried_section: str) -> tuple[list[_Unit], str, list[str]]:
    """Split one page on shape-detected headings.

    Returns the units, the section to carry into the next page, and the headings
    found (surfaced during ingest so a missed heading is visible rather than
    silent). Carrying the section forward is what stops a section split across a
    page break from losing its title — placement_policy page 2 opens mid-sentence
    inside §3.1 and must not be labelled as a new section.
    """
    units: list[_Unit] = []
    section = carried_section
    found: list[str] = []
    current: list[str] = []

    for line in doc.text.split("\n"):
        if is_pdf_heading(line):
            if "\n".join(current).strip():
                units.append(_Unit("\n".join(current), section, doc.page))
                current = []
            section = line.strip()
            found.append(section)
        else:
            current.append(line)

    if "\n".join(current).strip():
        units.append(_Unit("\n".join(current), section, doc.page))
    return units, section, found


def _split_yaml(doc: RawDoc) -> list[_Unit]:
    """One unit per rendered record — these are already chunk-sized (§b.4)."""
    return [_Unit(doc.text, str(doc.extra.get("record", "")), doc.page, atomic=True)]


# --------------------------------------------------------------------------
# sizing
# --------------------------------------------------------------------------


def _window(text: str, max_chars: int, overlap: int) -> list[str]:
    """Paragraph-aligned sliding window, used only when a unit is oversized.

    Packs whole paragraphs up to the limit, then starts the next window with the
    tail of the previous one so a fact spanning the seam is recoverable from
    either side. A single paragraph longer than the limit is split on sentence
    boundaries rather than mid-word.
    """
    parts = [p for p in _PARA_SPLIT.split(text) if p.strip()]
    if len(parts) <= 1:
        parts = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()] or [text]

    expanded: list[str] = []
    for part in parts:
        if len(part) <= max_chars:
            expanded.append(part)
        else:
            expanded.extend(s for s in _SENTENCE_SPLIT.split(part) if s.strip())

    out: list[str] = []
    current = ""
    for part in expanded:
        if current and len(current) + len(part) + 2 > max_chars:
            out.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{part}" if tail else part
        else:
            current = f"{current}\n\n{part}" if current else part
    if current.strip():
        out.append(current)
    return out


def _apply_sizing(units: list[_Unit], s: Settings) -> list[_Unit]:
    """Force-split oversized units, then merge runts forward."""
    sized: list[_Unit] = []
    for unit in units:
        if len(unit) <= s.max_chars:
            sized.append(unit)
            continue
        for piece in _window(unit.text, s.max_chars, s.overlap_chars):
            sized.append(_Unit(piece, unit.section, unit.page, unit.atomic))

    # Runts merge FORWARD into the following unit (§b.3): a short unit is
    # usually a lead-in to what comes next, so attaching it to its successor
    # preserves reading order. Atomic units (FAQ pairs, YAML records) are
    # indexed as-is however short, and a runt is never merged across a section
    # or page boundary — that would blur two topics to save a few characters.
    merged: list[_Unit] = []
    pending: Optional[_Unit] = None
    for unit in sized:
        if pending is not None:
            if (
                not unit.atomic
                and pending.section == unit.section
                and pending.page == unit.page
                and len(pending) + len(unit) + 2 <= s.max_chars
            ):
                unit = _Unit(f"{pending.text}\n\n{unit.text}", unit.section, unit.page)
            else:
                merged.append(pending)  # nothing suitable to merge into
            pending = None

        if not unit.atomic and len(unit) < s.min_chars:
            pending = unit
        else:
            merged.append(unit)

    if pending is not None:
        merged.append(pending)
    return merged


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------


def breadcrumb(doc_title: str, doc_id: str, section: str, page: Optional[int]) -> str:
    """The §b.2 header, prepended to indexed text and excluded from the body."""
    parts = [p for p in (doc_title, doc_id) if p]
    if section:
        parts.append(f"§{section}")
    if page is not None:
        parts.append(f"page {page}")
    return f"[{' | '.join(parts)}]"


def _chunk_authority(doc: RawDoc, defers_to: tuple[str, ...]) -> Authority:
    """Per-chunk authority (§c.6).

    A FAQ chunk that names another document is a summary of that document — the
    FAQ's own refund section says "quote that document, not this summary". Every
    other chunk inherits its document's authority.
    """
    if doc.doc_type == "faq" and defers_to:
        return "summary"
    return doc.authority


def chunk_documents(
    docs: Iterable[RawDoc], settings: Settings, verbose: bool = False
) -> list[Chunk]:
    """Turn loaded documents into indexed, citable chunks."""
    chunks: list[Chunk] = []
    counters: dict[str, int] = {}
    carried: dict[str, str] = {}  # source_file -> section carried across pages

    for doc in docs:
        if doc.doc_type == "faq":
            units = _split_faq(doc)
        elif doc.doc_type == "yaml":
            units = _split_yaml(doc)
        elif doc.doc_type == "pdf":
            units, carried[doc.source_file], found = _split_pdf(
                doc, carried.get(doc.source_file, "")
            )
            if verbose and found:
                print(f"    {doc.source_file} p{doc.page}: {' / '.join(found)}")
        else:
            units = _split_text(doc)

        for unit in _apply_sizing(units, settings):
            body = unit.text.strip()
            if not body:
                continue

            # Chunk-level deference: IDs named in THIS chunk, union the
            # document-level set. A chunk that names no other document still
            # inherits its document's deference.
            own_refs = tuple(
                r for r in dict.fromkeys(DOC_ID_RE.findall(body)) if r != doc.doc_id
            )
            defers_to = tuple(dict.fromkeys(own_refs + doc.defers_to))

            n = counters.get(doc.source_file, 0)
            counters[doc.source_file] = n + 1

            header = breadcrumb(doc.doc_title, doc.doc_id, unit.section, unit.page)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.source_file}::{n}",
                    text=f"{header}\n{body}",
                    body=body,
                    source_file=doc.source_file,
                    doc_title=doc.doc_title,
                    doc_id=doc.doc_id,
                    doc_type=doc.doc_type,
                    section=unit.section,
                    authority=_chunk_authority(doc, defers_to),
                    defers_to=defers_to,
                    # Hash the body, not the indexed text: a change to the
                    # breadcrumb format must not invalidate every embedding.
                    content_hash=hashlib.sha256(body.encode("utf-8")).hexdigest(),
                    page=unit.page,
                    char_len=len(body),
                )
            )
    return chunks


def summarise(chunks: list[Chunk]) -> str:
    """Distribution report, checked against §b.3's expected 140-180 chunks."""
    if not chunks:
        return "no chunks produced"

    lengths = sorted(c.char_len for c in chunks)
    per_doc: dict[str, list[int]] = {}
    for c in chunks:
        per_doc.setdefault(c.source_file, []).append(c.char_len)

    lines = [
        f"{len(chunks)} chunks | "
        f"chars min={lengths[0]} median={int(statistics.median(lengths))} "
        f"max={lengths[-1]} mean={int(statistics.mean(lengths))}",
        "",
        f"  {'document':<26}{'chunks':>7}{'median':>8}{'max':>7}",
        f"  {'-' * 46}",
    ]
    for doc, ls in sorted(per_doc.items()):
        lines.append(
            f"  {doc:<26}{len(ls):>7}{int(statistics.median(ls)):>8}{max(ls):>7}"
        )

    authority_counts: dict[str, int] = {}
    for c in chunks:
        authority_counts[c.authority] = authority_counts.get(c.authority, 0) + 1
    lines += ["", "  authority: " + ", ".join(f"{k}={v}" for k, v in sorted(authority_counts.items()))]

    # The measured baseline for this corpus is 183 (DESIGN.md §b.3). The band is
    # a tripwire for a splitter regression — a broken heading regex collapses the
    # count, a broken banner match explodes it — not a target to tune toward.
    if not 165 <= len(chunks) <= 205:
        lines += [
            "",
            f"  WARNING: {len(chunks)} chunks is outside the 165-205 band DESIGN.md §b.3",
            "  records for this corpus (measured baseline: 183). A large swing usually",
            "  means a structural splitter stopped matching. Check before ingesting.",
        ]
    return "\n".join(lines)
