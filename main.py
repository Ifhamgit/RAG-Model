"""Meridian Academy RAG — command line entry point.

    python main.py --ingest [--force]     build the index from corpus/

Later steps add --search, --query, --traces, and --serve.
"""

from __future__ import annotations

import argparse
import logging
import sys

from rag.config import settings
from rag.embeddings import EmbeddingBackendError
from rag.llm import LLMError


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(name)-16s %(message)s",
        stream=sys.stderr,
    )
    # fastembed's ONNX provider chatter is noise at INFO.
    logging.getLogger("fastembed").setLevel(logging.WARNING)


def cmd_ingest(args: argparse.Namespace) -> int:
    from rag.pipeline import ingest

    print(f"Ingesting from {settings.corpus_dir}")
    print("  detecting headings:")
    report = ingest(settings, force=args.force)
    print()
    print(report.render())
    print()
    print(f"  index: {settings.db_path}")
    print(f"         {settings.vectors_path}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    """Retrieval-only inspection. No LLM involved.

    This is the tool DESIGN.md §e.2 step 3 assumes exists: comparing dense_rank
    against bm25_rank is how a retrieval failure is localised to one arm.
    """
    from rag.embeddings import get_embedder
    from rag.retriever import Retriever
    from rag.store import Store

    with Store(settings.db_path, settings.vectors_path) as store:
        if store.count() == 0:
            print("error: index is empty. Run `python main.py --ingest` first.", file=sys.stderr)
            return 2
        r = Retriever(store, get_embedder(settings), settings)
        res = r.retrieve(args.search)

    print(f'\nQ: "{res.query}"')
    if res.expanded_query:
        print(f"   expanded: {res.expanded_query}")
    print()
    head = (
        f"{'#':<3}{'doc':<25}{'section':<34}{'pg':>3} {'auth':<14}"
        f"{'dns':>5}{'cos':>7}{'bm':>4}{'bm25':>8}{'rrf':>9}  def"
    )
    print(head)
    print("-" * len(head))
    for c in res.chunks:
        k = c.chunk
        dr = str(c.dense_rank) if c.dense_rank else "-"
        ds = f"{c.dense_score:.3f}" if c.dense_score is not None else "-"
        br = str(c.bm25_rank) if c.bm25_rank else "-"
        bs = f"{c.bm25_score:.3f}" if c.bm25_score is not None else "-"
        mark = "*" if c.authority_demoted else " "
        print(
            f"{c.final_rank:<3}{k.source_file:<25}{k.section[:33]:<34}"
            f"{(k.page if k.page else ''):>3} {k.authority:<14}"
            f"{dr:>5}{ds:>7}{br:>4}{bs:>8}{c.rrf_score:>8.5f}{mark} "
            f"{','.join(d.split('-')[1] for d in k.defers_to) or '-'}"
        )
        print(f"     {k.body[:150].replace(chr(10), ' ')}")
    if not res.chunks:
        print("  (no results)")

    print(f"\n  * = demoted by the §c.6 authority rule (x{settings.authority_penalty})")
    print(
        f"  top_score={res.top_score if res.top_score is None else round(res.top_score, 4)}"
        f"  score_gap={res.score_gap if res.score_gap is None else round(res.score_gap, 4)}"
        f"  top_bm25={res.top_bm25 if res.top_bm25 is None else round(res.top_bm25, 3)}"
    )
    print(
        f"  dense_miss={res.dense_miss} (floor {settings.miss_dense_cosine})"
        f"  sparse_miss={res.sparse_miss} (floor {settings.miss_bm25_floor})"
        f"  HARD_MISS={res.hard_miss}"
    )
    print(f"  content_terms={res.content_terms}")
    print(f"  matched_terms={res.matched_terms}")
    print(
        f"  latency: embed {res.latency_embed_ms:.1f} ms, retrieve "
        f"{res.latency_retrieve_ms:.1f} ms"
    )
    return 0


def cmd_check_llm(_args: argparse.Namespace) -> int:
    """One trivial round-trip, to prove the key and provider resolve correctly."""
    from rag.llm import LLMClient

    info = settings.describe()
    print("configuration")
    for k in ("provider", "model", "base_url", "effort", "api_key_present"):
        print(f"  {k:<16} {info[k]}")

    client = LLMClient(settings)

    print("\nplain completion")
    r = client.complete(
        system="You are a terse assistant. Reply with exactly one word.",
        user="Reply with the single word: ready",
        max_tokens=64,
        effort="low",
    )
    print(f"  reply            {r.text.strip()[:60]!r}")
    print(f"  model returned   {r.model}")
    print(f"  tokens           in={r.input_tokens} out={r.output_tokens}")
    print(f"  latency          {r.latency_ms:.0f} ms")
    print(f"  finish_reason    {r.finish_reason}")

    print("\nstructured output (the answerer depends on this)")
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "citations": {"type": "array", "items": {"type": "string"}},
            "sufficient_context": {"type": "boolean"},
        },
        "required": ["answer", "citations", "sufficient_context"],
        "additionalProperties": False,
    }
    r2 = client.complete(
        system="Answer only from the SOURCES. Cite the bracket IDs you used.",
        user='SOURCES:\n\n[S1] The refund on day 45 is 40% of tuition and GST is not '
             'refunded.\n\nQUESTION: What is the refund on day 45?',
        schema=schema,
        max_tokens=512,
        effort="low",
    )
    print(f"  parsed           {r2.parsed}")
    print(f"  tokens           in={r2.input_tokens} out={r2.output_tokens}")
    print(f"  latency          {r2.latency_ms:.0f} ms")
    if r2.parsed is None:
        print("\n  WARNING: structured output did not parse. Raw text follows:")
        print("  " + r2.text[:400].replace("\n", "\n  "))
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Grounded, cited Q&A over the Meridian Academy corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See DESIGN.md for the design and its reasoning.",
    )
    p.add_argument("--ingest", action="store_true", help="build the index from corpus/")
    p.add_argument(
        "--force",
        action="store_true",
        help="with --ingest: wipe and rebuild rather than reusing unchanged embeddings",
    )
    p.add_argument(
        "--search",
        metavar="QUERY",
        help="retrieval only, no LLM: show the ranked chunks with per-arm scores and ranks",
    )
    p.add_argument(
        "--check-llm",
        action="store_true",
        help="one round-trip to verify the key, provider and structured output",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    if not args.ingest and not args.search and not args.check_llm:
        parser.print_help()
        return 1

    # Every failure below is one a user can act on, so it is reported as a
    # message rather than a traceback. --verbose still shows the stack.
    try:
        if args.ingest:
            return cmd_ingest(args)
        if args.check_llm:
            return cmd_check_llm(args)
        return cmd_search(args)
    except FileNotFoundError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 2
    except NotADirectoryError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 2
    except EmbeddingBackendError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 3
    except LLMError as exc:
        print(f"\nerror [{exc.stage}]: {exc}", file=sys.stderr)
        return 4
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # pragma: no cover - last-resort guard
        if args.verbose:
            raise
        print(f"\nunexpected error: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("re-run with --verbose for the traceback", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
