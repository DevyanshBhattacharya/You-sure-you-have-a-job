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

from sqlalchemy import func, select  # noqa: E402

from app.agent import llm  # noqa: E402
from app.db import init_db, session_scope  # noqa: E402
from app.kb import indexer  # noqa: E402
from app.models import ApplicationEvent, Email, KBChunk  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report only, change nothing")
    parser.add_argument("--limit", type=int, default=0, help="Cap emails processed (0 = all)")
    args = parser.parse_args()

    init_db()

    if not llm.is_configured():
        print("No LLM backend configured; nothing to embed with.")
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

        print(f"\n{len(emails)} job-related email(s) to re-index.")
        if args.dry_run:
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
