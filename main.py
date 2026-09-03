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
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    if not args.ingest:
        parser.print_help()
        return 1

    # Every failure below is one a user can act on, so it is reported as a
    # message rather than a traceback. --verbose still shows the stack.
    try:
        return cmd_ingest(args)
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
