"""Re-embed the knowledge base with the currently configured model.

Needed after switching providers or embedding models: vectors of different
dimensions are not comparable, so old chunks become invisible to search until
they are rebuilt.

    python scripts/reindex_kb.py --dry-run
    python scripts/reindex_kb.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import delete, func, select  # noqa: E402

from app.agent import llm  # noqa: E402
from app.db import init_db, session_scope  # noqa: E402
from app.kb import indexer  # noqa: E402
from app.kb.store import store  # noqa: E402
from app.models import ApplicationEvent, Email, KBChunk  # noqa: E402


def prune(session, *, dry_run: bool) -> int:
    """Delete chunks belonging to emails that are no longer job related.

    Indexing only ever ran on the way in, so an email later re-classified out of
    the job search kept its chunks and stayed searchable — the Q&A agent would
    quote a job-board digest as evidence. Needs no model, so it is safe to run
    when the backend is down.
    """
    orphans = session.scalars(
        select(KBChunk.id)
        .join(Email, Email.id == KBChunk.email_id, isouter=True)
        .where((Email.id.is_(None)) | (Email.is_job_related.is_not(True)))
    ).all()

    if not orphans:
        print("No stale chunks to prune.")
        return 0

    print(f"Pruning {len(orphans)} chunk(s) from mail that is no longer job related.")
    if dry_run:
        return len(orphans)

    session.execute(delete(KBChunk).where(KBChunk.id.in_(orphans)))
    session.flush()
    store.invalidate()
    return len(orphans)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report only, change nothing")
    parser.add_argument("--limit", type=int, default=0, help="Cap emails processed (0 = all)")
    parser.add_argument(
        "--prune-only",
        action="store_true",
        help="Only drop chunks for mail that is no longer job related. Needs no model.",
    )
    args = parser.parse_args()

    init_db()

    if args.prune_only:
        with session_scope() as session:
            prune(session, dry_run=args.dry_run)
            if args.dry_run:
                session.rollback()
                print("Dry run - nothing written.")
        return 0

    if not llm.is_configured():
        print("No LLM backend configured; nothing to embed with.")
        print("Run with --prune-only to clean stale chunks without one.")
        return 1

    described = llm.describe()
    print(f"Provider : {described.get('provider')}")
    print(f"Embedding: {described.get('embedding')}\n")

    with session_scope() as session:
        by_dim = session.execute(
            select(KBChunk.dim, func.count()).group_by(KBChunk.dim)
        ).all()
        print("Existing chunks by dimension:")
        for dim, count in by_dim:
            print(f"  dim={dim if dim is not None else 'unembedded':>12}  {count}")

        emails = session.scalars(
            select(Email).where(Email.is_job_related.is_(True)).order_by(Email.received_at.desc())
        ).all()
        if args.limit:
            emails = emails[: args.limit]

        print()
        prune(session, dry_run=args.dry_run)

        print(f"\n{len(emails)} job-related email(s) to re-index.")
        if args.dry_run:
            session.rollback()
            print("Dry run - nothing written.")
            return 0

        done = 0
        for email in emails:
            # Keep chunks attached to the application the email belongs to.
            event = session.scalar(
                select(ApplicationEvent).where(ApplicationEvent.email_id == email.id)
            )
            try:
                indexer.index_email(
                    session, email, application_id=event.application_id if event else None
                )
            except llm.DeferWorkError as exc:
                print(f"\nStopped: backend unavailable ({exc}). Re-run later to continue.")
                break
            except Exception as exc:  # noqa: BLE001
                print(f"  failed on email {email.id}: {str(exc)[:120]}")
                continue

            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(emails)}")

        print(f"\nRe-indexed {done} email(s).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
