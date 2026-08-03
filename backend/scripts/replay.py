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
import builtins
import functools
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

    # Output is usually redirected to a file, where Python block-buffers it and
    # a long run looks like a hung one for minutes at a time.
    print = functools.partial(builtins.print, flush=True)  # noqa: A001

    with session_scope() as session:
        emails = select_emails(session, args)
        ids = [e.id for e in emails]
        if not ids:
            print("No emails matched.")
            return 0

    print(f"Replaying {len(ids)} email(s){' (dry run)' if args.dry_run else ''}\n")

    changed = 0
    failed = 0

    for index, email_id in enumerate(ids, start=1):
        # One session per email, committed as it goes. A single run can take
        # tens of minutes on a local model, and wrapping the whole loop in one
        # transaction meant one slow message threw away every verdict before
        # it — which is a long wait to repeat for no reason.
        try:
            with session_scope() as session:
                email = session.get(Email, email_id)
                if email is None:
                    continue

                before = email.is_job_related
                before_source = email.classification_source
                subject = (email.subject or "(no subject)")[:50]

                if args.dry_run:
                    verdict = classify_mod.classify(session, email)
                    after = verdict.is_job_related
                    confidence = verdict.confidence
                    session.rollback()
                else:
                    result = pipeline.process_email(session, email, reclassify=True)
                    after = result["email"]["is_job_related"]
                    confidence = result["email"]["confidence"]
                    retracted = len(result.get("retracted_applications") or [])
                    if retracted:
                        subject = f"{subject}  [-{retracted} application]"
        except Exception as exc:  # noqa: BLE001 - report and keep going
            failed += 1
            print(f"[FAILED ] {index}/{len(ids)} {str(exc)[:110]}")
            continue

        if before != after:
            changed += 1
            marker = "CHANGED"
        else:
            marker = "same   "

        print(
            f"[{marker}] {before!s:>5} -> {after!s:<5} "
            f"({confidence:.2f} via {before_source or 'new'}) {subject}"
        )

    print(f"\n{changed} of {len(ids)} verdict(s) changed; {failed} failed.")
    if args.dry_run:
        print("Dry run - nothing was written.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
