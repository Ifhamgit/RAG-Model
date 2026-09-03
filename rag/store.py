"""Store — SQLite + FTS5 + a NumPy matrix (DESIGN.md §c.2).

No vector database, deliberately. At ~183 chunks a brute-force `matrix @ qvec`
scan is tens of microseconds; an ANN index would be *slower* here once you count
its build, and it adds a service to run and a schema to keep in sync with the
vectors for no measurable gain. Choosing infrastructure the data does not need
is a real engineering error, not a safe default.

Three things fall out of that choice, and they are the reason it is the right
one at this size:

  * FTS5 is built into SQLite, so the BM25 arm costs nothing extra and lives in
    the same file and the same transaction as the chunks.
  * Traces live here too, so investigating a bad answer is one SQL join from
    trace to retrieved chunk to chunk text (§e.2).
  * The whole index is two files you can delete and rebuild in seconds.

The public surface is deliberately narrow — add_chunks / dense_search /
bm25_search / get_chunks / count / meta / write_trace / reset — so swapping in
pgvector later is a single-file change. It stops being the right choice past
roughly 100k chunks, on concurrent writers, or when the matrix no longer fits
in RAM.
"""

from __future__ import annotations

import json
import logging
import pathlib
import re
import sqlite3
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from .models import Chunk

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      TEXT PRIMARY KEY,
    text          TEXT NOT NULL,   -- indexed text: breadcrumb header + body
    body          TEXT NOT NULL,   -- bare content: what the model is shown
    source_file   TEXT NOT NULL,
    doc_title     TEXT NOT NULL,
    doc_id        TEXT NOT NULL,
    doc_type      TEXT NOT NULL,
    section       TEXT NOT NULL,
    page          INTEGER,
    authority     TEXT NOT NULL,
    defers_to     TEXT NOT NULL DEFAULT '[]',
    char_len      INTEGER NOT NULL,
    content_hash  TEXT NOT NULL,
    vector_row    INTEGER
);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_file);
CREATE INDEX IF NOT EXISTS idx_chunks_hash   ON chunks(content_hash);

-- External-content FTS5: the index stores only the inverted index and reads
-- the text back from `chunks`, so the corpus is not duplicated on disk.
-- `porter` wraps unicode61 with Porter stemming, and it is load-bearing rather
-- than a refinement. Without it FTS5 matches tokens exactly: the corpus says
-- "Refunds" and "refunded", a user types "refund", and the sparse arm returns
-- nothing for the single most common phrasing in this corpus. Measured on a
-- fixture chunk: "refund" -> 0 hits unstemmed, 1 hit stemmed.
-- Changing this string requires rebuilding the index (`--ingest --force`);
-- CREATE ... IF NOT EXISTS will not alter an existing table.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='rowid',
    tokenize='porter unicode61 remove_diacritics 2'
);

-- External-content tables are not maintained automatically; without these
-- triggers the BM25 arm silently returns stale or deleted rows.
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.rowid, old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.rowid, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- One row per query (§e.1). Same database as chunks so a failure investigation
-- is a join, not an export.
CREATE TABLE IF NOT EXISTS traces (
    trace_id              TEXT PRIMARY KEY,
    timestamp_utc         TEXT NOT NULL,
    query                 TEXT NOT NULL,
    expanded_query        TEXT,
    retrieved             TEXT NOT NULL DEFAULT '[]',  -- JSON, per-arm scores AND ranks
    retrieval_mode        TEXT,
    embedding_backend     TEXT,
    prompt_token_estimate INTEGER,
    prompt_char_len       INTEGER,
    prompt_sha256         TEXT,
    prompt_full           TEXT,
    n_sources             INTEGER,
    answer                TEXT,
    citations             TEXT DEFAULT '[]',
    sufficient_context    INTEGER,
    reasoning_note        TEXT,
    citation_integrity    TEXT,
    invalid_citations     TEXT DEFAULT '[]',
    unsupported_claim     INTEGER,
    latency_ms            REAL,
    latency_expand_ms     REAL,
    latency_embed_ms      REAL,
    latency_retrieve_ms   REAL,
    latency_llm_ms        REAL,
    latency_verify_ms     REAL,
    input_tokens          INTEGER,
    output_tokens         INTEGER,
    model                 TEXT,
    est_cost_usd          REAL,
    no_context            INTEGER,
    escalate              INTEGER,
    top_score             REAL,
    score_gap             REAL,
    error                 TEXT,
    error_stage           TEXT
);
CREATE INDEX IF NOT EXISTS idx_traces_ts ON traces(timestamp_utc);
"""

TRACE_COLUMNS = tuple(
    re.findall(r"^\s{4}(\w+)\s+(?:TEXT|INTEGER|REAL)", SCHEMA.split("CREATE TABLE IF NOT EXISTS traces")[1], re.M)
)

# FTS5 treats these as query syntax. A learner's question is not a query
# expression, so they are stripped rather than escaped.
_FTS_UNSAFE = re.compile(r'[^\w\s]', re.UNICODE)


class Store:
    """Chunks, their vectors, the keyword index, and the traces."""

    def __init__(self, db_path: pathlib.Path, vectors_path: pathlib.Path):
        self.db_path = pathlib.Path(db_path)
        self.vectors_path = pathlib.Path(vectors_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

        self._matrix: Optional[np.ndarray] = None
        self._rows: Optional[list[str]] = None  # vector_row -> chunk_id

    # ------------------------------------------------------------------ io
    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def reset(self) -> None:
        """Drop all indexed content. Traces survive — they are the record of what
        the system did, and a rebuild is not a reason to lose it."""
        self.conn.execute("DELETE FROM chunks")
        self.conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        self.conn.execute("DELETE FROM meta")
        self.conn.commit()
        self.vectors_path.unlink(missing_ok=True)
        self._matrix = None
        self._rows = None

    # -------------------------------------------------------------- writes
    def add_chunks(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        """Persist chunks and their vectors as one unit.

        The two must agree: `chunks[i]` is described by `vectors[i]`, and that
        correspondence is recorded as `vector_row` so a later reordering of the
        table cannot silently detach a chunk from its embedding.
        """
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")

        rows = [
            (
                c.chunk_id, c.text, c.body, c.source_file, c.doc_title, c.doc_id,
                c.doc_type, c.section, c.page, c.authority, json.dumps(list(c.defers_to)),
                c.char_len, c.content_hash, i,
            )
            for i, c in enumerate(chunks)
        ]
        with self.conn:
            self.conn.execute("DELETE FROM chunks")
            self.conn.executemany(
                "INSERT INTO chunks (chunk_id, text, body, source_file, doc_title, doc_id,"
                " doc_type, section, page, authority, defers_to, char_len, content_hash,"
                " vector_row) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )

        matrix = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
        self.vectors_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.vectors_path, matrix)
        self._matrix = matrix
        self._rows = [c.chunk_id for c in chunks]

    def set_meta(self, key: str, value: Any) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(value)),
            )

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return default if row is None else json.loads(row["value"])

    def all_meta(self) -> dict[str, Any]:
        return {r["key"]: json.loads(r["value"]) for r in self.conn.execute("SELECT * FROM meta")}

    # --------------------------------------------------------------- reads
    def count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"])

    def get_chunks(self, chunk_ids: Iterable[str]) -> dict[str, Chunk]:
        ids = list(chunk_ids)
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})", ids
        ).fetchall()
        return {r["chunk_id"]: _row_to_chunk(r) for r in rows}

    def all_chunks(self) -> list[Chunk]:
        rows = self.conn.execute("SELECT * FROM chunks ORDER BY vector_row").fetchall()
        return [_row_to_chunk(r) for r in rows]

    def content_hashes(self) -> dict[str, str]:
        """chunk_id -> content_hash, for the ingest idempotency check."""
        return {
            r["chunk_id"]: r["content_hash"]
            for r in self.conn.execute("SELECT chunk_id, content_hash FROM chunks")
        }

    # ------------------------------------------------------------ dense arm
    def _load_matrix(self) -> tuple[np.ndarray, list[str]]:
        """Load once and cache. Loading per query would dominate the runtime."""
        if self._matrix is None or self._rows is None:
            if not self.vectors_path.exists():
                raise FileNotFoundError(
                    f"No vector index at {self.vectors_path}. Run `python main.py --ingest` first."
                )
            self._matrix = np.load(self.vectors_path).astype(np.float32)
            rows = self.conn.execute(
                "SELECT chunk_id FROM chunks ORDER BY vector_row"
            ).fetchall()
            self._rows = [r["chunk_id"] for r in rows]
            if len(self._rows) != self._matrix.shape[0]:
                raise RuntimeError(
                    f"Index is inconsistent: {len(self._rows)} chunks but "
                    f"{self._matrix.shape[0]} vectors. Rebuild with `--ingest --force`."
                )
        return self._matrix, self._rows

    def dense_search(self, qvec: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Cosine similarity over the whole matrix.

        Every embedder L2-normalises, so a dot product *is* cosine — one BLAS
        call, no per-row normalisation, no distance-to-similarity conversion.
        """
        matrix, rows = self._load_matrix()
        if matrix.size == 0:
            return []
        scores = matrix @ np.asarray(qvec, dtype=np.float32)
        k = min(k, scores.shape[0])
        # argpartition finds the top-k without sorting all 183; the slice is
        # then sorted properly. Immaterial at this size, correct at any size.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(rows[i], float(scores[i])) for i in top]

    # ----------------------------------------------------------- sparse arm
    @staticmethod
    def to_fts_query(query: str) -> str:
        """Turn a natural-language question into a safe FTS5 MATCH expression.

        FTS5 gives `"`, `*`, `^`, `-`, `NEAR`, `AND`/`OR`/`NOT` syntactic
        meaning. A learner's question contains apostrophes and question marks,
        not query operators, so punctuation is stripped and each token is
        double-quoted as a literal. Unquoted, `can't` raises "fts5: syntax
        error" and the sparse arm dies on an ordinary question.
        """
        tokens = [t for t in _FTS_UNSAFE.sub(" ", query).split() if t]
        # Bare AND/OR/NOT would be parsed as operators even after quoting removes
        # the risk elsewhere; dropping them costs nothing, they are stopwords.
        tokens = [t for t in tokens if t.upper() not in {"AND", "OR", "NOT", "NEAR"}]
        return " OR ".join(f'"{t}"' for t in tokens)

    def bm25_search(self, query: str, k: int) -> list[tuple[str, float]]:
        """BM25 over the chunk text.

        SQLite's `bm25()` returns *negative* scores, where more negative means a
        better match, so it can be used directly in `ORDER BY`. Every other
        score in this codebase is higher-is-better, so it is negated here — at
        the boundary — rather than leaving a sign convention for the retriever
        and the traces to remember.
        """
        match = self.to_fts_query(query)
        if not match:
            return []
        try:
            rows = self.conn.execute(
                "SELECT c.chunk_id AS chunk_id, bm25(chunks_fts) AS score "
                "FROM chunks_fts JOIN chunks c ON c.rowid = chunks_fts.rowid "
                "WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
                (match, k),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            # A malformed MATCH must not take down a query; the dense arm can
            # still answer, and the empty sparse result is itself a miss signal.
            # But log it — silently returning [] here once hid a real bug, and a
            # dead sparse arm looks identical to a legitimate no-match.
            log.warning("FTS5 MATCH failed for %r (expr=%r): %s", query, match, exc)
            return []
        return [(r["chunk_id"], -float(r["score"])) for r in rows]

    # -------------------------------------------------------------- traces
    def write_trace(self, row: dict[str, Any]) -> None:
        """Insert one trace, coercing types SQLite cannot store directly."""
        payload = {k: row.get(k) for k in TRACE_COLUMNS}
        for key, value in payload.items():
            if isinstance(value, bool):
                payload[key] = int(value)
            elif isinstance(value, (list, dict, tuple)):
                payload[key] = json.dumps(value, default=str)
        cols = ",".join(payload)
        marks = ",".join("?" * len(payload))
        with self.conn:
            self.conn.execute(
                f"INSERT OR REPLACE INTO traces ({cols}) VALUES ({marks})",
                tuple(payload.values()),
            )

    def recent_traces(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM traces ORDER BY timestamp_utc DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_trace(self, trace_id: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute("SELECT * FROM traces WHERE trace_id=?", (trace_id,)).fetchone()
        return dict(row) if row else None


def _row_to_chunk(r: sqlite3.Row) -> Chunk:
    return Chunk(
        chunk_id=r["chunk_id"],
        text=r["text"],
        body=r["body"],
        source_file=r["source_file"],
        doc_title=r["doc_title"],
        doc_id=r["doc_id"],
        doc_type=r["doc_type"],
        section=r["section"],
        authority=r["authority"],
        defers_to=tuple(json.loads(r["defers_to"])),
        content_hash=r["content_hash"],
        page=r["page"],
        char_len=r["char_len"],
    )
