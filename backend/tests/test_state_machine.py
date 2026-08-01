"""Status transition rules.

The point of the state machine is that a late-arriving or misclassified email
can never walk an application backwards.
"""

from __future__ import annotations

import pytest

from app.agent.resolve import next_status
from app.models import ApplicationStatus as S
from app.models import EventType as E


@pytest.mark.parametrize(
    ("current", "event", "expected"),
    [
        # Forward progress
        (S.DISCOVERED, E.APPLIED, S.APPLIED),
        (S.APPLIED, E.ACKNOWLEDGEMENT, S.ACKNOWLEDGED),
        (S.ACKNOWLEDGED, E.SCREEN_SCHEDULED, S.SCREENING),
        (S.SCREENING, E.ASSESSMENT_SENT, S.ASSESSMENT),
        (S.ASSESSMENT, E.INTERVIEW_SCHEDULED, S.INTERVIEWING),
        (S.INTERVIEWING, E.OFFER, S.OFFER),
        # Skipping stages is allowed — recruiters don't follow our taxonomy
        (S.APPLIED, E.INTERVIEW_SCHEDULED, S.INTERVIEWING),
        (S.DISCOVERED, E.OFFER, S.OFFER),
    ],
)
def test_forward_transitions(current, event, expected):
    assert next_status(current, event) is expected


@pytest.mark.parametrize(
    ("current", "event"),
    [
        # A late "thanks for applying" auto-reply must not undo real progress.
        (S.OFFER, E.APPLIED),
        (S.INTERVIEWING, E.ACKNOWLEDGEMENT),
        (S.ASSESSMENT, E.APPLIED),
        (S.SCREENING, E.RECRUITER_OUTREACH),
    ],
)
def test_backwards_transitions_are_blocked(current, event):
    assert next_status(current, event) is current


@pytest.mark.parametrize(
    ("current", "event", "expected"),
    [
        (S.APPLIED, E.REJECTION, S.REJECTED),
        (S.INTERVIEWING, E.REJECTION, S.REJECTED),
        (S.OFFER, E.REJECTION, S.REJECTED),
        (S.SCREENING, E.WITHDRAWN, S.WITHDRAWN),
    ],
)
def test_terminal_states_are_reachable_from_anywhere(current, event, expected):
    assert next_status(current, event) is expected


@pytest.mark.parametrize(
    ("current", "event"),
    [
        (S.REJECTED, E.INTERVIEW_SCHEDULED),
        (S.REJECTED, E.ACKNOWLEDGEMENT),
        (S.ACCEPTED, E.REJECTION),
        (S.WITHDRAWN, E.OFFER),
    ],
)
def test_declared_outcomes_are_sticky(current, event):
    """Once rejected/accepted/withdrawn, stray mail can't reopen it.

    A genuine re-engagement is almost always a different role, and so becomes
    its own application.
    """
    assert next_status(current, event) is current


def test_ghosted_is_recoverable():
    """`ghosted` is inferred, not declared, so real activity overrides it."""
    assert next_status(S.GHOSTED, E.INTERVIEW_SCHEDULED) is S.INTERVIEWING
    assert next_status(S.GHOSTED, E.REJECTION) is S.REJECTED


@pytest.mark.parametrize("event", [E.FOLLOW_UP, E.OTHER])
def test_neutral_events_never_change_status(event):
    for status in S:
        assert next_status(status, event) is status
