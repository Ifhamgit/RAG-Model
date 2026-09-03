"""Evaluation harness — all five metrics (DESIGN.md §f).

Answer quality is the product of two independently-failing stages, so retrieval
and generation are measured separately before they are measured together. If
end-to-end accuracy drops, the component metrics say which half broke.

    python eval/run_eval.py --retrieval-only   no LLM at all: recall@k. Free.
    python eval/run_eval.py --no-judge         answers generated, judges skipped
    python eval/run_eval.py                    all five metrics
    python eval/run_eval.py --case <id>        one case

Exits non-zero if any case fails, so it works as a CI gate.

Read §f.3 before trusting any number here. The short version: n=12 is a smoke
test, the questions were written by the system's author from the corpus, and the
judge shares a model family with the system under test.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rag.answerer import build_context  # noqa: E402
from rag.config import settings  # noqa: E402
from rag.embeddings import get_embedder  # noqa: E402
from rag.pipeline import QueryEngine  # noqa: E402
from rag.retriever import Retriever  # noqa: E402
from rag.store import Store  # noqa: E402
from rag.tracing import estimate_cost_usd  # noqa: E402

CASES_PATH = pathlib.Path(__file__).parent / "test_cases.json"
RESULTS_DIR = pathlib.Path(__file__).parent / "results"

# Thresholds a case must clear to pass. Deliberately strict: these are the
# properties the design claims, so weakening them to make the suite green would
# make the suite worthless.
FAITHFULNESS_MIN = 0.80
CORRECTNESS_MIN = 1.0  # every required fact must be present


def load_cases() -> list[dict]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]


# ---------------------------------------------------------------- metric 1
def recall_at_k(case: dict, retrieved) -> tuple[bool, str]:
    """Deterministic, LLM-free. Generation quality is capped by this, so it is
    the first thing to check on any regression."""
    exp = case.get("expected_source")
    if exp is None:
        return True, "n/a (abstention case)"
    files = [r.chunk.source_file for r in retrieved]
    if exp["file"] not in files:
        return False, f"expected {exp['file']}, got {files}"
    rank = files.index(exp["file"]) + 1
    detail = f"{exp['file']} @{rank}"
    want = exp.get("section_contains")
    if want:
        hit = any(
            r.chunk.source_file == exp["file"] and want.lower() in r.chunk.section.lower()
            for r in retrieved
        )
        detail += f" §~{'y' if hit else 'n'}"
    return True, detail


def run_case(case: dict, engine: QueryEngine, judge, mode: str) -> dict:
    """One case end to end. Returns a row with every metric that `mode` allows."""
    row: dict = {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "expect_abstain": case["expect_abstain"],
        "expect_hard_miss": case.get("expect_hard_miss", False),
    }

    if mode == "retrieval-only":
        res = engine.retriever.retrieve(case["question"])
        ok, detail = recall_at_k(case, res.chunks)
        row.update(_retrieval_fields(res, ok, detail))
        return row

    r = engine.ask(case["question"])
    ok, detail = recall_at_k(case, r.sources)
    row.update({
        "recall_ok": ok,
        "recall_detail": detail,
        "answer": r.answer,
        "citations": r.citations,
        "sufficient_context": r.sufficient_context,
        "escalate": r.escalate,
        "no_context": r.no_context,
        "hard_miss": r.no_context,
        "citation_integrity": r.citation_integrity,
        "invalid_citations": r.invalid_citations,
        "unsupported_claim": r.unsupported_claim,
        "top_score": r.top_score,
        "score_gap": r.score_gap,
        "latency_ms": r.latency_ms,
        "input_tokens": r.input_tokens,
        "output_tokens": r.output_tokens,
        "est_cost_usd": estimate_cost_usd(r.model, r.input_tokens, r.output_tokens),
        "trace_id": r.trace_id,
        "retrieved": [
            {"chunk_id": s.chunk.chunk_id, "doc": s.chunk.source_file,
             "section": s.chunk.section, "authority": s.chunk.authority,
             "dense_rank": s.dense_rank, "bm25_rank": s.bm25_rank,
             "demoted": s.authority_demoted}
            for s in r.sources
        ],
    })

    # ------------------------------------------------------------ metric 5
    # Deterministic: did the system abstain exactly when it should have?
    row["abstain_ok"] = (not r.sufficient_context) == case["expect_abstain"]

    if judge is None:
        return row

    context, _ = build_context(r.sources)

    # ------------------------------------------------------------ metric 2
    if case["expect_abstain"]:
        row["correctness"] = {"score": 1.0, "verdicts": [], "note": "abstention case"}
    else:
        row["correctness"] = judge.correctness(
            case["question"], r.answer, case.get("expected_facts", [])
        )

    # ------------------------------------------------------------ metric 3
    row["faithfulness"] = judge.faithfulness(r.answer, context)

    # ------------------------------------------------------------ metric 4
    row["context_precision"] = judge.context_precision(
        case["question"], [(f"S{i}", s.chunk.body) for i, s in enumerate(r.sources, 1)]
    )
    return row


def _retrieval_fields(res, ok: bool, detail: str) -> dict:
    return {
        "recall_ok": ok, "recall_detail": detail,
        "top_score": res.top_score, "score_gap": res.score_gap,
        "dense_miss": res.dense_miss, "sparse_miss": res.sparse_miss,
        "hard_miss": res.hard_miss,
        "content_terms": res.content_terms, "matched_terms": res.matched_terms,
    }


def verdict(row: dict, mode: str) -> tuple[bool, list[str]]:
    """A case passes only if every metric it ran clears its bar."""
    fails: list[str] = []
    if not row.get("recall_ok", True):
        fails.append("recall")
    if mode == "retrieval-only":
        if row.get("hard_miss") != row.get("expect_hard_miss"):
            fails.append("hard_miss")
        return not fails, fails

    if not row.get("abstain_ok", True):
        fails.append("abstain")
    if row.get("citation_integrity") == "violated":
        fails.append("citations")
    if row.get("unsupported_claim"):
        fails.append("unsupported")

    c = row.get("correctness")
    if c and c["score"] < CORRECTNESS_MIN:
        missing = [v["fact"] for v in c.get("verdicts", []) if not v.get("present")]
        fails.append(f"correctness({','.join(m[:22] for m in missing)})")

    f = row.get("faithfulness")
    if f:
        # Any contradicted claim fails outright: the answer states something the
        # context denies, which is the exact failure this system exists to avoid.
        if f.get("contradicted"):
            fails.append("faithfulness(contradicted)")
        elif f["score"] < FAITHFULNESS_MIN:
            fails.append(f"faithfulness({f['score']:.2f})")
    return not fails, fails


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate the RAG pipeline (DESIGN.md §f).")
    ap.add_argument("--retrieval-only", action="store_true",
                    help="no LLM at all: recall@k and miss routing. Free.")
    ap.add_argument("--no-judge", action="store_true",
                    help="generate answers but skip the three judged metrics")
    ap.add_argument("--case", metavar="ID", help="run a single case")
    ap.add_argument("--workers", type=int, default=4, help="parallel cases (default 4)")
    ap.add_argument("--no-save", action="store_true", help="do not write eval/results/")
    args = ap.parse_args(argv)

    cases = load_cases()
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            print(f"error: no case with id {args.case!r}", file=sys.stderr)
            return 2

    mode = "retrieval-only" if args.retrieval_only else ("no-judge" if args.no_judge else "full")
    t_start = time.perf_counter()

    if mode == "retrieval-only":
        with Store(settings.db_path, settings.vectors_path) as store:
            if store.count() == 0:
                print("error: index is empty. Run `python main.py --ingest`.", file=sys.stderr)
                return 2

            class _E:  # minimal shim: retrieval-only needs no LLM client
                retriever = Retriever(store, get_embedder(settings), settings)

            rows = [run_case(c, _E, None, mode) for c in cases]
    else:
        with QueryEngine(settings) as engine:
            if engine.store.count() == 0:
                print("error: index is empty. Run `python main.py --ingest`.", file=sys.stderr)
                return 2
            judge = None
            if mode == "full":
                from judge import Judge  # noqa: E402  (eval/ is on sys.path via __file__)

                judge = Judge(engine.client)

            # Cases are independent, and each is dominated by network latency —
            # so they run in parallel. The LLM client's backoff has jitter so a
            # shared rate limit does not cause synchronised retries.
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(run_case, c, engine, judge, mode): c for c in cases}
                rows = []
                for fut in concurrent.futures.as_completed(futures):
                    c = futures[fut]
                    try:
                        rows.append(fut.result())
                    except Exception as exc:
                        rows.append({
                            "id": c["id"], "category": c["category"],
                            "question": c["question"],
                            "expect_abstain": c["expect_abstain"],
                            "expect_hard_miss": c.get("expect_hard_miss", False),
                            "error": f"{type(exc).__name__}: {exc}",
                            "recall_ok": False,
                        })
            order = {c["id"]: i for i, c in enumerate(cases)}
            rows.sort(key=lambda r: order[r["id"]])

    # ------------------------------------------------------------- report
    for r in rows:
        r["passed"], r["failures"] = verdict(r, mode)

    print(f"\n{'=' * 92}")
    print(f"EVALUATION — {len(rows)} cases, mode={mode}, "
          f"{time.perf_counter() - t_start:.0f}s")
    print("=" * 92)

    if mode == "retrieval-only":
        print(f"\n  {'':<5}{'case':<24}{'top_cos':>8}{'dMiss':>7}{'sMiss':>7}{'hard':>6}  detail")
        for r in rows:
            print(f"  {'ok' if r['passed'] else 'FAIL':<5}{r['id']:<24}"
                  f"{_n(r['top_score']):>8}{str(r['dense_miss']):>7}{str(r['sparse_miss']):>7}"
                  f"{str(r['hard_miss']):>6}  {r['recall_detail'][:34]}")
    else:
        hdr = (f"  {'':<5}{'case':<24}{'rec':>5}{'abs':>5}{'corr':>7}{'faith':>7}"
               f"{'prec':>7}{'ms':>7}  failures")
        print()
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for r in rows:
            if r.get("error"):
                print(f"  {'ERR':<5}{r['id']:<24}  {r['error'][:56]}")
                continue
            print(f"  {'ok' if r['passed'] else 'FAIL':<5}{r['id']:<24}"
                  f"{'y' if r['recall_ok'] else 'n':>5}"
                  f"{'y' if r['abstain_ok'] else 'n':>5}"
                  f"{_pct(r.get('correctness')):>7}{_pct(r.get('faithfulness')):>7}"
                  f"{_pct(r.get('context_precision')):>7}{r['latency_ms']:>7.0f}"
                  f"  {', '.join(r['failures'])[:34]}")

    # ---------------------------------------------------------- aggregates
    ok_rows = [r for r in rows if not r.get("error")]
    answerable = [r for r in ok_rows if not r["expect_abstain"]]
    print("\nMETRICS")
    print(f"  1. recall@{settings.top_k:<19} {_mean([r['recall_ok'] for r in answerable]):>6}"
          f"   over {len(answerable)} answerable cases")
    if mode != "retrieval-only":
        print(f"  5. abstention accuracy      {_mean([r['abstain_ok'] for r in ok_rows]):>6}"
              f"   over all {len(ok_rows)} cases")
    if mode == "full":
        print(f"  2. answer correctness       "
              f"{_mean([r['correctness']['score'] for r in ok_rows if 'correctness' in r]):>6}")
        print(f"  3. faithfulness             "
              f"{_mean([r['faithfulness']['score'] for r in ok_rows if 'faithfulness' in r]):>6}"
              f"   contradicted claims: "
              f"{sum(r['faithfulness'].get('contradicted', 0) for r in ok_rows if 'faithfulness' in r)}")
        print(f"  4. context precision        "
              f"{_mean([r['context_precision']['score'] for r in ok_rows if 'context_precision' in r]):>6}")

    if mode != "retrieval-only" and ok_rows:
        lat = sorted(r["latency_ms"] for r in ok_rows)
        toks = [r["input_tokens"] + r["output_tokens"] for r in ok_rows]
        cost = [r["est_cost_usd"] for r in ok_rows]
        tops = [r["top_score"] for r in ok_rows if r["top_score"] is not None]
        gaps = [r["score_gap"] for r in ok_rows if r["score_gap"] is not None]
        print("\nALSO REPORTED (not pass/fail — a quality gain paid for with 10x latency")
        print("is a trade-off someone should get to make explicitly)")
        print(f"  latency p50 / p95         {_pctl(lat, 50):.0f} / {_pctl(lat, 95):.0f} ms")
        print(f"  tokens mean               {statistics.mean(toks):.0f}")
        print(f"  est cost / query          ${statistics.mean(cost):.5f}"
              f"   (${sum(cost):.4f} for this run, judge calls excluded)")
        print(f"  top_score mean            {statistics.mean(tops):.4f}" if tops else "")
        print(f"  score_gap mean            {statistics.mean(gaps):.4f}" if gaps else "")

    failed = [r for r in rows if not r["passed"]]
    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / f"{time.strftime('%Y%m%dT%H%M%S')}-{mode}.json"
        out.write_text(json.dumps({
            "mode": mode,
            "settings": settings.describe(),
            "elapsed_s": round(time.perf_counter() - t_start, 1),
            "rows": rows,
        }, indent=2, default=str), encoding="utf-8")
        print(f"\n  full results incl. every judge justification: {out}")

    print()
    if failed:
        print(f"FAILED {len(failed)}/{len(rows)}: "
              + "; ".join(f"{r['id']} [{', '.join(r['failures'])}]" for r in failed))
        return 1
    print(f"PASSED all {len(rows)} cases")
    return 0


def _n(v, nd: int = 3) -> str:
    return "-" if v is None else f"{float(v):.{nd}f}"


def _pct(m) -> str:
    return "-" if not m else f"{m['score'] * 100:.0f}%"


def _mean(xs) -> str:
    xs = list(xs)
    return "-" if not xs else f"{statistics.mean(float(x) for x in xs) * 100:.0f}%"


def _pctl(sorted_xs: list[float], p: int) -> float:
    if not sorted_xs:
        return 0.0
    k = (len(sorted_xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(sorted_xs) - 1)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (k - lo)


if __name__ == "__main__":
    raise SystemExit(main())
