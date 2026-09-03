"""Loaders — get text out of whatever this is, and normalise early (DESIGN.md §b.4).

Four file types, one output shape. After this boundary nothing downstream knows
or cares that `course_brochure.pdf` was a PDF, which is what keeps the chunker
free of per-format branches.

The other job of this module is stamping **document authority** (§c.6). The
corpus is deliberately cross-referential — the FAQ summarises the refund policy
and says in its own text not to quote it — so "which document governs" is
metadata that has to be captured at load time or it is lost.
"""

from __future__ import annotations

import pathlib
import re
from typing import Any, Protocol, runtime_checkable

import yaml
from pypdf import PdfReader

from .models import Authority, RawDoc

# Document IDs as they actually appear in this corpus: MA-REFUND-2026-01,
# MA-PRICING-2026-03, and — the reason this is not the narrower `MA-...` pattern
# BUILD_PLAN suggested — SUP-FAQ-2026-03. The FAQ uses a different prefix, so a
# rule anchored on "MA-" would silently fail to extract the ID of the one
# document whose authority handling matters most.
DOC_ID_RE = re.compile(r"\b[A-Z]{2,5}-[A-Z]{2,12}-\d{4}-\d{2}\b")

# Self-declared authority. Both corpus phrasings are covered: refund_terms says
# "Status: AUTHORITATIVE...", placement_policy says "Status: Authoritative for
# all placement matters", program_pricing says "single source of truth".
AUTHORITY_MARKERS = ("authoritative", "single source of truth")

# Lines that name a document ID for reasons that are NOT deference. A
# "Supersedes:" line points at a retired predecessor; treating that as deference
# would demote the current document in favour of a version that no longer
# exists.
NON_DEFERENCE_PREFIXES = ("supersedes:", "version:", "supersedes ")

_HEADER_LINES = 14  # how far into a document to look for its own header block


# --------------------------------------------------------------------------
# text normalisation
# --------------------------------------------------------------------------

_DEHYPHENATE = re.compile(r"(\w)-\n(\w)")
_MULTI_BLANK = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+\n")


def normalise(text: str) -> str:
    """Clean extracted text without destroying the structure the chunker needs.

    Non-breaking spaces are converted to ordinary spaces before anything else.
    The rendered tables in the PDFs use them for column alignment, and FTS5
    treats U+00A0 as a word character — so left alone, a figure like "6,40,000"
    sitting in an nbsp-padded table column never becomes a searchable token and
    the BM25 arm cannot match it (§b.4).

    Blank lines are preserved: they are the paragraph boundaries the chunker's
    sliding window aligns to.
    """
    text = text.replace(" ", " ").replace("​", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _DEHYPHENATE.sub(r"\1\2", text)  # rejoin words split across a line
    text = _TRAILING_WS.sub("\n", text)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------
# header parsing — the single place authority is decided
# --------------------------------------------------------------------------


def parse_header(text: str, source_file: str, body: str | None = None) -> dict[str, Any]:
    """Extract doc_title, doc_id, authority, and defers_to from a document's own header.

    `text` is the header region — page 1 of a PDF, the leading comment block of
    the YAML, the top of a text file. `body` is the *whole* document and
    defaults to `text`. The two are separate because a document's deference can
    appear far from its header: the brochure names the documents it defers to in
    a "Next steps" section on its last page, so scanning only page 1 would
    classify it as authoritative — the opposite of what §c.6 requires.

    Kept in one function rather than spread across the loaders so that "how does
    this system decide what governs" has exactly one answer to read.

    Authority (§c.6), in precedence order:
      1. The header self-declares -> `authoritative`.
      2. Otherwise, the document points at other documents for the binding
         version -> `reference`. This is the brochure ("Next steps" names the
         pricing, eligibility and placement documents) and the FAQ.
      3. Otherwise -> `authoritative`, because a document that neither declares
         nor defers is the sole source for its own content. This is
         eligibility_criteria.txt, which has no status line but is the only
         place its rules exist.

    `defers_to` is always the explicit set of document IDs found in the body,
    minus this document's own. Nothing infers "overlapping topic" from the text
    — by design (§c.6). Only an ID the author actually wrote counts.
    """
    header_lines = [ln.strip() for ln in text.split("\n")]
    header_block = "\n".join(header_lines[:_HEADER_LINES])
    lines = [ln.strip() for ln in (body if body is not None else text).split("\n")]

    # Title: the first meaningful line, with YAML comment or markdown markers
    # stripped. Every document in the corpus opens with its own title.
    doc_title = ""
    for ln in header_lines:
        candidate = ln.lstrip("#").strip()
        if candidate:
            doc_title = candidate
            break

    header_ids = DOC_ID_RE.findall(header_block)
    doc_id = header_ids[0] if header_ids else ""

    # Body references, excluding this document's own ID and any line where the
    # reference is bookkeeping rather than deference.
    referenced: list[str] = []
    for ln in lines:
        low = ln.lower()
        if any(low.startswith(p) for p in NON_DEFERENCE_PREFIXES):
            continue
        for found in DOC_ID_RE.findall(ln):
            if found != doc_id and found not in referenced:
                referenced.append(found)

    lowered_header = header_block.lower()
    if any(marker in lowered_header for marker in AUTHORITY_MARKERS):
        authority: Authority = "authoritative"
    elif referenced:
        authority = "reference"
    else:
        authority = "authoritative"

    return {
        "doc_title": doc_title,
        "doc_id": doc_id or source_file,
        "authority": authority,
        "defers_to": tuple(referenced),
    }


# --------------------------------------------------------------------------
# loader protocol
# --------------------------------------------------------------------------


@runtime_checkable
class Loader(Protocol):
    """One file type in, a uniform list of RawDoc out."""

    def matches(self, path: pathlib.Path) -> bool: ...

    def load(self, path: pathlib.Path) -> list[RawDoc]: ...


class PdfLoader:
    """One RawDoc per page, so page numbers survive as citation anchors (§b.4)."""

    def matches(self, path: pathlib.Path) -> bool:
        return path.suffix.lower() == ".pdf"

    def load(self, path: pathlib.Path) -> list[RawDoc]:
        reader = PdfReader(str(path))
        pages = [normalise(page.extract_text() or "") for page in reader.pages]

        # The header lives on page 1, but its authority verdict applies to every
        # page — and the deference that decides that verdict may be recorded
        # anywhere in the document, so the whole text is scanned for references.
        meta = parse_header(
            pages[0] if pages else "", path.name, body="\n".join(pages)
        )

        return [
            RawDoc(
                text=page_text,
                source_file=path.name,
                doc_type="pdf",
                page=page_no,
                extra={"n_pages": len(pages)},
                **meta,
            )
            for page_no, page_text in enumerate(pages, start=1)
            if page_text.strip()
        ]


class FaqLoader:
    """faq.txt. One RawDoc; the Q&A splitting is the chunker's job (§b.4)."""

    def matches(self, path: pathlib.Path) -> bool:
        return path.name.lower() == "faq.txt"

    def load(self, path: pathlib.Path) -> list[RawDoc]:
        text = normalise(path.read_text(encoding="utf-8"))
        meta = parse_header(text, path.name)
        return [RawDoc(text=text, source_file=path.name, doc_type="faq", **meta)]


class TextLoader:
    """Plain policy text — refund_terms.txt, eligibility_criteria.txt."""

    def matches(self, path: pathlib.Path) -> bool:
        return path.suffix.lower() in {".txt", ".md"}

    def load(self, path: pathlib.Path) -> list[RawDoc]:
        text = normalise(path.read_text(encoding="utf-8"))
        meta = parse_header(text, path.name)
        return [RawDoc(text=text, source_file=path.name, doc_type="text", **meta)]


class YamlLoader:
    """program_pricing.yaml, rendered to prose (§b.4).

    Raw YAML is poor retrieval input: `list_tuition_inr: 210000` shares no
    tokens with "how much does the foundations course cost", so neither arm can
    find it. Each record is therefore flattened into sentence-shaped text, with
    every number reproduced verbatim so BM25 can still match "210000". The
    original fragment is kept in `extra` for exactness.
    """

    def matches(self, path: pathlib.Path) -> bool:
        return path.suffix.lower() in {".yaml", ".yml"}

    def load(self, path: pathlib.Path) -> list[RawDoc]:
        raw = path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw) or {}

        # The header is the leading comment block, which YAML parsing discards.
        comment_block = "\n".join(
            ln.lstrip("#").strip() for ln in raw.split("\n") if ln.lstrip().startswith("#")
        )
        meta = parse_header(comment_block, path.name, body=raw)

        docs: list[RawDoc] = []
        for key, value in data.items():
            # `programs` is a list of records; each one is its own retrievable
            # unit, because a question is almost always about one program.
            if key == "programs" and isinstance(value, list):
                for record in value:
                    code = record.get("code", "?")
                    docs.append(
                        RawDoc(
                            text=_render_program(record),
                            source_file=path.name,
                            doc_type="yaml",
                            extra={"record": code, "yaml": yaml.safe_dump(record, sort_keys=False)},
                            **meta,
                        )
                    )
            else:
                docs.append(
                    RawDoc(
                        text=f"{_humanise(key)}.\n{_render_value(value)}",
                        source_file=path.name,
                        doc_type="yaml",
                        extra={"record": key, "yaml": yaml.safe_dump({key: value}, sort_keys=False)},
                        **meta,
                    )
                )
        return docs


# --------------------------------------------------------------------------
# YAML -> prose
# --------------------------------------------------------------------------

# Tokens that must not be title-cased into meaninglessness when a snake_case key
# becomes prose.
_ACRONYMS = {
    "inr": "INR", "usd": "USD", "gst": "GST", "emi": "EMI", "nbfc": "NBFC",
    "mat": "MAT", "ctc": "CTC", "isa": "ISA", "id": "ID", "prg": "PRG",
}


def _humanise(key: str) -> str:
    words = [_ACRONYMS.get(w, w) for w in str(key).split("_")]
    out = " ".join(words)
    return out[:1].upper() + out[1:] if out else out


def _scalar(value: Any) -> str:
    """Render a leaf value. YAML folded scalars (`>`) keep a trailing newline,
    which would otherwise emit a stray "." on its own line."""
    return str(value).strip()


def _render_value(value: Any, indent: int = 0) -> str:
    """Walk a YAML subtree into readable lines, numbers preserved verbatim."""
    pad = "  " * indent
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            if isinstance(v, (dict, list)):
                parts.append(f"{pad}{_humanise(k)}:\n{_render_value(v, indent + 1)}")
            else:
                parts.append(f"{pad}{_humanise(k)}: {_scalar(v)}.")
        return "\n".join(parts)
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, (dict, list)):
                parts.append(_render_value(item, indent + 1))
            else:
                parts.append(f"{pad}- {_scalar(item)}")
        return "\n".join(parts)
    return f"{pad}{_scalar(value)}"


def _render_program(record: dict[str, Any]) -> str:
    """The §b.4 worked example shape, for the records users ask about most."""
    code = record.get("code", "")
    name = record.get("name", "")
    lead = f"Program {code}, {name}."

    rest = [
        f"{_humanise(k)}: {_scalar(v)}."
        for k, v in record.items()
        if k not in {"code", "name"} and not isinstance(v, (dict, list))
    ]
    nested = [
        f"{_humanise(k)}:\n{_render_value(v, 1)}"
        for k, v in record.items()
        if isinstance(v, (dict, list))
    ]
    return "\n".join([lead, *rest, *nested])


# --------------------------------------------------------------------------
# corpus entry point
# --------------------------------------------------------------------------

LOADERS: tuple[Loader, ...] = (PdfLoader(), FaqLoader(), TextLoader(), YamlLoader())


def load_corpus(corpus_dir: pathlib.Path) -> list[RawDoc]:
    """Load every supported document, sorted by filename for determinism.

    Determinism matters beyond tidiness: chunk IDs are positional, so a stable
    file order means a re-ingest of an unchanged corpus produces byte-identical
    chunk IDs and the content-hash skip in the pipeline actually fires.
    """
    corpus_dir = pathlib.Path(corpus_dir)
    if not corpus_dir.exists():
        raise FileNotFoundError(
            f"Corpus directory not found: {corpus_dir}\n"
            "Set CORPUS_DIR in .env, or create the directory and add the documents."
        )
    if not corpus_dir.is_dir():
        raise NotADirectoryError(f"CORPUS_DIR is not a directory: {corpus_dir}")

    docs: list[RawDoc] = []
    skipped: list[str] = []
    for path in sorted(corpus_dir.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_file() or path.name.startswith("."):
            continue
        for loader in LOADERS:
            if loader.matches(path):
                docs.extend(loader.load(path))
                break
        else:
            skipped.append(path.name)

    if not docs:
        raise ValueError(
            f"No loadable documents in {corpus_dir}. "
            f"Supported types: .pdf, .txt, .md, .yaml, .yml."
            + (f" Ignored: {', '.join(skipped)}." if skipped else "")
        )
    return docs
