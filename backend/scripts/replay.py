"""Re-classify stored emails without touching Gmail.

The point of this script is prompt iteration: change the classifier prompt or
schema, replay against mail you already have, and see exactly which verdicts
moved — without burning Gmail quota or waiting on a re-sync.

    python scripts/replay.py --dry-run          # show diffs, change nothing
    python scripts/replay.py --limit 50         # apply to 50 emails
    python scripts/replay.py --only-job-related # re-check known positives
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.agent import classify as classify_mod  # noqa: E402
from app.agent import pipeline  # noqa: E402
from app.db import init_db, session_scope  # noqa: E402
from app.models import Email  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=100, help="Max emails to replay")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Classify and report, but don't write anything back",
    )
    parser.add_argument(
        "--only-job-related",
        action="store_true",
        help="Only replay emails currently marked job-related",
    )
    parser.add_argument(
        "--only-unprocessed",
        action="store_true",
        help="Only replay emails that were never classified",
    )
    parser.add_argument("--gmail-id", help="Replay one specific message")
    return parser.parse_args()


def select_emails(session, args: argparse.Namespace) -> list[Email]:
    stmt = select(Email)
    if args.gmail_id:
        stmt = stmt.where(Email.gmail_id == args.gmail_id)
    else:
        if args.only_job_related:
            stmt = stmt.where(Email.is_job_related.is_(True))
        if args.only_unprocessed:
            stmt = stmt.where(Email.processed_at.is_(None))
        stmt = stmt.order_by(Email.received_at.desc()).limit(args.limit)
    return list(session.scalars(stmt).all())


def main() -> int:
    args = parse_args()
    init_db()

    with session_scope() as session:
        emails = select_emails(session, args)
        if not emails:
            print("No emails matched.")
            return 0

        print(f"Replaying {len(emails)} email(s){' (dry run)' if args.dry_run else ''}\n")

        changed = 0
        for email in emails:
            before = email.is_job_related
            before_source = email.classification_source

            if args.dry_run:
                verdict = classify_mod.classify(session, email)
                after = verdict.is_job_related
                confidence = verdict.confidence
            else:
                result = pipeline.process_email(session, email, reclassify=True)
                after = result["email"]["is_job_related"]
                confidence = result["email"]["confidence"]

            if before != after:
                changed += 1
                marker = "CHANGED"
            else:
                marker = "same   "

            subject = (email.subject or "(no subject)")[:50]
            print(
                f"[{marker}] {before!s:>5} -> {after!s:<5} "
                f"({confidence:.2f} via {before_source or 'new'}) {subject}"
            )

        if args.dry_run:
            session.rollback()

        print(f"\n{changed} of {len(emails)} verdict(s) changed.")
        if args.dry_run:
            print("Dry run - nothing was written.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
