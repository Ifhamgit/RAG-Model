"""Evaluation harness (DESIGN.md §f).

Answer quality is the product of two independently-failing stages, so retrieval
and generation are measured separately before they are measured together. If
end-to-end accuracy drops, the component metrics say which half broke.

    python eval/run_eval.py --no-judge     deterministic only: no LLM, free, fast
    python eval/run_eval.py                all five metrics (adds the LLM judge)
    python eval/run_eval.py --case <id>    one case

Step 7 implements the deterministic half — Recall@k and the calibration report.
The three judged metrics arrive with the answerer in Step 12; the CLI already
accepts their flags so the interface does not change under you later.

Exits non-zero if any case fails, so it works as a CI gate.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rag.config import settings  # noqa: E402
from rag.embeddings import get_embedder  # noqa: E402
from rag.retriever import Retriever  # noqa: E402
from rag.store import Store  # noqa: E402

CASES_PATH = pathlib.Path(__file__).parent / "test_cases.json"
RESULTS_DIR = pathlib.Path(__file__).parent / "results"


def load_cases() -> list[dict]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]


def recall_at_k(case: dict, retrieved) -> tuple[bool, str]:
    """Did the expected source document appear in the top-k?

    Deterministic and LLM-free, so it is the first thing to check on any
    regression: generation quality is capped by it. Section matching is a
    substring test and is reported but not required — the document is the
    contract, the section is a hint about precision.
    """
    exp = case.get("expected_source")
    if exp is None:  # abstention case: recall is not meaningful
        return True, "n/a (abstention case)"

    files = [r.chunk.source_file for r in retrieved]
    if exp["file"] not in files:
        return False, f"expected {exp['file']}, got {files}"

    rank = files.index(exp["file"]) + 1
    detail = f"{exp['file']} at rank {rank}"

    want_section = exp.get("section_contains")
    if want_section:
        hit = any(
            r.chunk.source_file == exp["file"] and want_section.lower() in r.chunk.section.lower()
            for r in retrieved
        )
        detail += f"; section~'{want_section}': {'yes' if hit else 'no'}"
    return True, detail


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate the RAG pipeline (DESIGN.md §f).")
    ap.add_argument("--no-judge", action="store_true",
                    help="deterministic metrics only: no LLM calls, no cost")
    ap.add_argument("--case", metavar="ID", help="run a single case by id")
    ap.add_argument("--save", action="store_true", help="write full results to eval/results/")
    args = ap.parse_args(argv)

    cases = load_cases()
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            print(f"error: no case with id {args.case!r}", file=sys.stderr)
            return 2

    judged = not args.no_judge
    if judged:
        print("note: judged metrics (correctness, faithfulness, context precision) arrive")
        print("      with the answerer in Step 12. Running deterministic metrics only.\n")

    with Store(settings.db_path, settings.vectors_path) as store:
        if store.count() == 0:
            print("error: index is empty. Run `python main.py --ingest` first.", file=sys.stderr)
            return 2
        retriever = Retriever(store, get_embedder(settings), settings)

        rows = []
        for case in cases:
            res = retriever.retrieve(case["question"])
            ok, detail = recall_at_k(case, res.chunks)
            rows.append({
                "id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "expect_abstain": case["expect_abstain"],
                "expect_hard_miss": case.get("expect_hard_miss", False),
                "recall_ok": ok,
                "recall_detail": detail,
                "top_score": res.top_score,
                "score_gap": res.score_gap,
                "top_bm25": res.top_bm25,
                "dense_miss": res.dense_miss,
                "sparse_miss": res.sparse_miss,
                "hard_miss": res.hard_miss,
                "content_terms": res.content_terms,
                "matched_terms": res.matched_terms,
                "retrieved": [
                    {"chunk_id": r.chunk.chunk_id, "doc": r.chunk.source_file,
                     "section": r.chunk.section, "authority": r.chunk.authority,
                     "dense_rank": r.dense_rank, "bm25_rank": r.bm25_rank,
                     "rrf": round(r.rrf_score, 5), "demoted": r.authority_demoted}
                    for r in res.chunks
                ],
            })

    # ---------------------------------------------------------------- report
    print(f"RETRIEVAL — Recall@{settings.top_k}\n")
    hdr = (f"  {'':<3}{'case':<24}{'category':<32}{'top_cos':>8}{'dMiss':>7}"
           f"{'sMiss':>7}{'hard':>6}  detail")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        mark = "ok " if r["recall_ok"] else "FAIL"
        ts = f"{r['top_score']:.3f}" if r["top_score"] is not None else "-"
        print(f"  {mark:<3}{r['id']:<24}{r['category'][:31]:<32}{ts:>8}"
              f"{str(r['dense_miss']):>7}{str(r['sparse_miss']):>7}"
              f"{str(r['hard_miss']):>6}  {r['recall_detail'][:44]}")

    answerable = [r for r in rows if not r["expect_abstain"]]
    abstention = [r for r in rows if r["expect_abstain"]]
    recall = sum(r["recall_ok"] for r in answerable) / max(len(answerable), 1)

    print(f"\n  recall@{settings.top_k} over {len(answerable)} answerable cases: {recall:.0%}")

    # ------------------------------------------------- threshold calibration
    # §c.4 says MISS_DENSE_COSINE is set by inspecting top_score on the
    # abstention cases against the answerable ones and placing the floor between
    # the clusters. That data is only available once retrieval runs, so it is
    # computed here rather than guessed twice.
    ans_scores = sorted(r["top_score"] for r in answerable if r["top_score"] is not None)
    abs_scores = sorted(r["top_score"] for r in abstention if r["top_score"] is not None)
    print("\nCALIBRATION — top dense cosine (§c.4)\n")
    if ans_scores:
        print(f"  answerable  n={len(ans_scores)}  min={ans_scores[0]:.4f} "
              f"median={statistics.median(ans_scores):.4f} max={ans_scores[-1]:.4f}")
    if abs_scores:
        print(f"  abstention  n={len(abs_scores)}  min={abs_scores[0]:.4f} "
              f"median={statistics.median(abs_scores):.4f} max={abs_scores[-1]:.4f}")
    if ans_scores and abs_scores:
        lo, hi = max(abs_scores), min(ans_scores)
        if lo < hi:
            print(f"  separable: abstention max {lo:.4f} < answerable min {hi:.4f}")
            print(f"  -> MISS_DENSE_COSINE anywhere in ({lo:.4f}, {hi:.4f}); "
                  f"midpoint {(lo + hi) / 2:.4f}")
        else:
            print(f"  NOT separable by dense score alone: abstention max {lo:.4f} "
                  f">= answerable min {hi:.4f}")

    # Separability of the dense score alone is the wrong question, because
    # hard_miss is an AND. What actually matters is the margin among the cases
    # that already trip sparse_miss — everywhere else, a dense false positive is
    # vetoed by the sparse arm and costs nothing.
    sparse_hits = [r for r in rows if r["sparse_miss"]]
    print("\n  cases where sparse_miss already holds (the only ones the dense floor decides):")
    if not sparse_hits:
        print("    none")
    for r in sorted(sparse_hits, key=lambda r: r["top_score"] or 0):
        want = "SHOULD miss" if r["expect_abstain"] else "must NOT miss"
        print(f"    {r['top_score']:.4f}  {r['id']:<24} {want}")
    bad = [r for r in sparse_hits if not r["expect_abstain"] and r["top_score"] is not None]
    good = [r for r in sparse_hits if r["expect_abstain"] and r["top_score"] is not None]
    if good:
        ceiling = max(r["top_score"] for r in good)
        floor_ = min((r["top_score"] for r in bad), default=None)
        if floor_ is None:
            print(f"    -> MISS_DENSE_COSINE must exceed {ceiling:.4f}; no answerable case "
                  f"competes, so any value above it is safe.")
        else:
            print(f"    -> MISS_DENSE_COSINE in ({ceiling:.4f}, {floor_:.4f})")

    # Not every abstention case should hard-miss. §d.3 has three paths, and only
    # the first belongs to the retriever: a question whose words are simply not
    # in the corpus. A question built entirely from corpus vocabulary that is
    # nonetheless unanswerable ("the 2026 cohort", when the figures are 2025)
    # must reach the model — catching it at retrieval would mean inferring
    # intent from a date, which §c.6 deliberately refuses to do. Each case
    # declares which path it expects.
    print(f"\n  current MISS_DENSE_COSINE = {settings.miss_dense_cosine}")
    miss_failures = []
    for r in rows:
        want = bool(r.get("expect_hard_miss", False))
        if not r["expect_abstain"] and not want and not r["hard_miss"]:
            continue  # ordinary answerable case behaving correctly
        ok = r["hard_miss"] == want
        if not ok:
            miss_failures.append(r["id"])
        route = "retriever short-circuit" if want else "model must abstain (§d.3 weak retrieval)"
        print(f"  [{'ok ' if ok else 'FAIL'}] {r['id']:<24} expect hard_miss={want} "
              f"got {r['hard_miss']}  -> {route}")

    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        import time
        out = RESULTS_DIR / f"{time.strftime('%Y%m%dT%H%M%S')}.json"
        out.write_text(json.dumps({
            "settings": settings.describe(),
            "recall_at_k": recall,
            "rows": rows,
        }, indent=2, default=str), encoding="utf-8")
        print(f"\n  wrote {out}")

    failures = [r["id"] for r in rows if not r["recall_ok"]] + miss_failures
    print()
    if failures:
        print(f"FAILED: {len(set(failures))} case(s) — {', '.join(sorted(set(failures)))}")
        return 1
    print("PASSED: all retrieval checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
