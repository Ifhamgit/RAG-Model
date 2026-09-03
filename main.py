"""Meridian Academy RAG — command line entry point.

    python main.py --ingest [--force]     build the index from corpus/
    python main.py --search "..."         retrieval only, no LLM
    python main.py --query  "..."         ask a question, get a cited answer
    python main.py --check-llm            verify the key and provider resolve
    python main.py --traces [N]           recent query traces
    python main.py --trace <id>           one trace in full
    python main.py --serve                HTTP API + browser UI on :8000
"""

from __future__ import annotations

import argparse
import json
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


def cmd_query(args: argparse.Namespace) -> int:
    from rag.pipeline import QueryEngine

    with QueryEngine(settings) as engine:
        if engine.store.count() == 0:
            print("error: index is empty. Run `python main.py --ingest` first.", file=sys.stderr)
            return 2
        r = engine.ask(args.query)

    print(f'\nQ: {r.question}')
    print("\nANSWER")
    for line in _wrap(r.answer, 96):
        print(f"  {line}")

    if r.sources:
        print("\nSOURCES")
        cited = set(r.citations)
        for i, s in enumerate(r.sources, start=1):
            sid = f"S{i}"
            mark = "*" if sid in cited else " "
            c = s.chunk
            page = f" p{c.page}" if c.page is not None else ""
            print(f" {mark}[{sid}] {c.source_file}{page} — §{c.section[:56]}")
            print(f"       {c.doc_id} | authority: {c.authority}"
                  f"{' | defers to ' + ','.join(c.defers_to) if c.defers_to else ''}")
            print(f"       dense r{s.dense_rank}/{_f(s.dense_score)}  "
                  f"bm25 r{s.bm25_rank}/{_f(s.bm25_score)}  rrf {s.rrf_score:.5f}"
                  f"{'  (demoted)' if s.authority_demoted else ''}")
        print("\n  * = cited by the answer")

    print(f"\n  sufficient_context : {r.sufficient_context}")
    print(f"  escalate           : {r.escalate}")
    print(f"  no_context         : {r.no_context}")
    print(f"  citation_integrity : {r.citation_integrity}"
          f"{'  invalid=' + str(r.invalid_citations) if r.invalid_citations else ''}")
    print(f"  unsupported_claim  : {r.unsupported_claim}")
    if r.reasoning_note:
        print(f"  reasoning_note     : {r.reasoning_note[:110]}")
    print(f"  top_score={_f(r.top_score)}  score_gap={_f(r.score_gap)}")
    print(f"  latency {r.latency_ms:.0f} ms  (expand {r.latency_expand_ms:.0f}, "
          f"embed {r.latency_embed_ms:.0f}, retrieve {r.latency_retrieve_ms:.0f}, "
          f"llm {r.latency_llm_ms:.0f}, verify {r.latency_verify_ms:.0f})")
    print(f"  tokens in={r.input_tokens} out={r.output_tokens}  model={r.model}")
    print(f"  trace_id {r.trace_id}")
    return 0


def cmd_traces(args: argparse.Namespace) -> int:
    """The §e.2 debugging entry point: find the trace, then read it."""
    from rag.store import Store

    with Store(settings.db_path, settings.vectors_path) as store:
        rows = store.recent_traces(args.traces)
    if not rows:
        print("no traces yet")
        return 0
    hdr = (f"  {'timestamp':<21}{'trace_id':<38}{'ms':>7}{'esc':>5}{'cit':>5}"
           f"{'top':>7}  query")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for t in rows:
        print(f"  {str(t['timestamp_utc']):<21}{str(t['trace_id']):<38}"
              f"{(t['latency_ms'] or 0):>7.0f}{str(bool(t['escalate'])):>5}"
              f"{len(json.loads(t['citations'] or '[]')):>5}"
              f"{_f(t['top_score']):>7}  {str(t['query'])[:60]}")
    return 0


def cmd_trace(args: argparse.Namespace) -> int:
    from rag.store import Store

    with Store(settings.db_path, settings.vectors_path) as store:
        t = store.get_trace(args.trace)
    if t is None:
        print(f"no trace with id {args.trace}", file=sys.stderr)
        return 2

    for key in ("trace_id", "timestamp_utc", "query", "expanded_query", "retrieval_mode",
                "embedding_backend", "model", "n_sources", "prompt_token_estimate",
                "prompt_char_len", "prompt_sha256", "sufficient_context", "escalate",
                "no_context", "citation_integrity", "invalid_citations", "unsupported_claim",
                "citations", "reasoning_note", "top_score", "score_gap",
                "latency_ms", "latency_expand_ms", "latency_embed_ms", "latency_retrieve_ms",
                "latency_llm_ms", "latency_verify_ms", "input_tokens", "output_tokens",
                "est_cost_usd", "error", "error_stage"):
        print(f"  {key:<22} {t.get(key)}")

    print("\n  answer:")
    for line in _wrap(str(t.get("answer") or ""), 92):
        print(f"    {line}")

    print("\n  retrieved (per-arm ranks are how §e.2 step 3 localises a failure):")
    hd = (f"    {'#':<3}{'doc':<24}{'section':<30}{'dns':>5}{'cos':>7}"
          f"{'bm':>4}{'bm25':>8}{'rrf':>9}")
    print(hd)
    for r in json.loads(t.get("retrieved") or "[]"):
        print(f"    {r.get('final_rank'):<3}{str(r.get('doc'))[:23]:<24}"
              f"{str(r.get('section'))[:29]:<30}{str(r.get('dense_rank')):>5}"
              f"{_f(r.get('dense_score')):>7}{str(r.get('bm25_rank')):>4}"
              f"{_f(r.get('bm25_score')):>8}{(r.get('rrf_score') or 0):>9.5f}")
    if t.get("prompt_full"):
        print("\n  prompt (TRACE_FULL_PROMPT=1):")
        print("    " + str(t["prompt_full"])[:2000].replace("\n", "\n    "))
    return 0


def _f(v, nd: int = 3) -> str:
    return "-" if v is None else f"{float(v):.{nd}f}"


def _wrap(text: str, width: int) -> list[str]:
    import textwrap

    out: list[str] = []
    for para in (text or "").split("\n"):
        out.extend(textwrap.wrap(para, width) or [""])
    return out


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    print(f"Serving on http://{args.host}:{args.port}")
    print("  GET  /            query UI")
    print("  POST /ask         {\"question\": \"...\"}")
    print("  GET  /health      what actually loaded on this machine")
    print("  GET  /traces      recent query traces")
    uvicorn.run("rag.api:app", host=args.host, port=args.port, log_level="info")
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
    p.add_argument("--query", metavar="QUESTION", help="ask a question; returns a cited answer")
    p.add_argument(
        "--traces", nargs="?", type=int, const=20, metavar="N",
        help="list the N most recent query traces (default 20)",
    )
    p.add_argument("--trace", metavar="ID", help="print one trace in full")
    p.add_argument("--serve", action="store_true", help="run the HTTP API and query UI")
    p.add_argument("--host", default="127.0.0.1", help="with --serve (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=8000, help="with --serve (default 8000)")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    if not any([args.ingest, args.search, args.check_llm, args.query,
                args.traces is not None, args.trace, args.serve]):
        parser.print_help()
        return 1

    # Every failure below is one a user can act on, so it is reported as a
    # message rather than a traceback. --verbose still shows the stack.
    try:
        if args.ingest:
            return cmd_ingest(args)
        if args.check_llm:
            return cmd_check_llm(args)
        if args.serve:
            return cmd_serve(args)
        if args.query:
            return cmd_query(args)
        if args.traces is not None:
            return cmd_traces(args)
        if args.trace:
            return cmd_trace(args)
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
