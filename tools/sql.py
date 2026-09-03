"""Run read-only SQL against the index database from the command line.

The machine has no sqlite3 CLI, and quoting SQL inside `python -c` under
PowerShell is fragile, so this is the debugging entry point for DESIGN.md §e.2.

    python tools/sql.py --tables
    python tools/sql.py "select trace_id, query from traces order by timestamp_utc desc limit 5"
    python tools/sql.py -f query.sql
    python tools/sql.py --wide "select text from chunks where chunk_id = 'refund_terms.txt::13'"
    python tools/sql.py --json "select * from meta"

The connection is opened read-only (`mode=ro`), so nothing here can modify
the index or the traces.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sqlite3
import sys

DEFAULT_DB = pathlib.Path("index") / "rag.db"
CELL_WIDTH = 60


def _connect(db: pathlib.Path) -> sqlite3.Connection:
    if not db.exists():
        sys.exit(f"database not found: {db}  (run `python main.py --ingest` first)")
    return sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)


def _fmt(value, wide: bool) -> str:
    if value is None:
        return "NULL"
    s = str(value).replace("\n", "\\n")
    if not wide and len(s) > CELL_WIDTH:
        s = s[: CELL_WIDTH - 1] + "…"
    return s


def _print_table(cols: list[str], rows: list[tuple], wide: bool) -> None:
    if not rows:
        print("(0 rows)")
        return
    cells = [[_fmt(v, wide) for v in row] for row in rows]
    widths = [max(len(c), *(len(r[i]) for r in cells)) for i, c in enumerate(cols)]
    line = "  ".join(c.ljust(widths[i]) for i, c in enumerate(cols))
    print(line)
    print("  ".join("-" * w for w in widths))
    for r in cells:
        print("  ".join(r[i].ljust(widths[i]) for i in range(len(cols))))
    print(f"({len(rows)} row{'s' if len(rows) != 1 else ''})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sql", nargs="?", help="SQL statement to run")
    p.add_argument("-f", "--file", help="read the SQL from a file instead")
    p.add_argument("--db", default=str(DEFAULT_DB), help=f"database path (default {DEFAULT_DB})")
    p.add_argument("--tables", action="store_true", help="list tables with row counts and exit")
    p.add_argument("--schema", metavar="TABLE", help="print the columns of one table and exit")
    p.add_argument("--wide", action="store_true", help="do not truncate long cells")
    p.add_argument("--json", action="store_true", help="print rows as JSON objects, one per line")
    args = p.parse_args()

    conn = _connect(pathlib.Path(args.db))

    if args.tables:
        names = [
            n
            for (n,) in conn.execute(
                "select name from sqlite_master where type='table' and name not like 'sqlite_%' "
                "and name not like 'chunks_fts_%' order by name"
            )
        ]
        for n in names:
            count = conn.execute(f'select count(*) from "{n}"').fetchone()[0]
            print(f"{n:12} {count:>6} rows")
        return

    if args.schema:
        rows = conn.execute(f'pragma table_info("{args.schema}")').fetchall()
        if not rows:
            sys.exit(f"no such table: {args.schema}")
        for _, name, ctype, *_ in rows:
            print(f"{name:24} {ctype}")
        return

    if args.file:
        sql = pathlib.Path(args.file).read_text(encoding="utf-8")
    elif args.sql:
        sql = args.sql
    else:
        p.error("give a SQL statement, -f FILE, --tables, or --schema TABLE")

    try:
        cur = conn.execute(sql)
    except sqlite3.Error as e:
        sys.exit(f"SQL error: {e}")

    cols = [d[0] for d in cur.description] if cur.description else []
    rows = cur.fetchall()

    if args.json:
        for row in rows:
            print(json.dumps(dict(zip(cols, row)), ensure_ascii=False, default=str))
        return

    _print_table(cols, rows, args.wide)


if __name__ == "__main__":
    main()
