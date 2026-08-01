"""MIME normalisation."""

from __future__ import annotations

from datetime import UTC

from app.gmail.client import parse_message, strip_quoted
from tests.fixtures import raw_message


def test_prefers_plain_text_over_html():
    parsed = parse_message(
        raw_message(plain="Plain body wins", html="<p>HTML body loses</p>")
    )
    assert parsed.body_text == "Plain body wins"


def test_falls_back_to_html_when_no_plain_part():
    html = (
        "<html><head><style>p{}</style></head>"
        "<body><p>Interview on Tuesday</p></body></html>"
    )
    parsed = parse_message(raw_message(plain=None, html=html))
    assert "Interview on Tuesday" in parsed.body_text
    assert "style" not in parsed.body_text


def test_html_keeps_link_targets():
    parsed = parse_message(
        plain_free := raw_message(
            plain=None,
            html='<a href="https://cal.example.com/book">Book a slot</a>',
        )
    )
    assert plain_free is not None
    assert "Book a slot" in parsed.body_text
    assert "https://cal.example.com/book" in parsed.body_text


def test_headers_and_addresses_are_extracted():
    parsed = parse_message(
        raw_message(
            sender="Jane Doe <Jane.Doe@Acme.com>",
            subject="Interview invitation",
            extra_headers={"List-Unsubscribe": "<mailto:x@y.z>"},
        )
    )
    assert parsed.from_addr == "jane.doe@acme.com"  # normalised to lowercase
    assert parsed.from_name == "Jane Doe"
    assert parsed.subject == "Interview invitation"
    assert "list-unsubscribe" in parsed.headers


def test_internal_date_becomes_utc_datetime():
    parsed = parse_message(raw_message())
    assert parsed.received_at is not None
    assert parsed.received_at.tzinfo is not None
    assert parsed.received_at.astimezone(UTC).year == 2026


def test_attachment_parts_are_skipped():
    message = raw_message(plain="Body text")
    message["payload"] = {
        "mimeType": "multipart/mixed",
        "headers": message["payload"]["headers"],
        "parts": [
            {"mimeType": "text/plain", "body": {"data": message["payload"]["body"]["data"]}},
            {"mimeType": "application/pdf", "body": {"attachmentId": "att1", "size": 9999}},
        ],
    }
    parsed = parse_message(message)
    assert parsed.body_text == "Body text"


class TestStripQuoted:
    def test_removes_on_wrote_reply_chain(self):
        text = (
            "Thanks, Tuesday works for me.\n"
            "\n"
            "On Mon, 2 Mar 2026 at 09:30, Jane Doe <jane@acme.com> wrote:\n"
            "> Are you free Tuesday?\n"
            "> Jane\n"
        )
        assert strip_quoted(text) == "Thanks, Tuesday works for me."

    def test_removes_original_message_divider(self):
        text = "Sure thing.\n\n-----Original Message-----\nFrom: someone\nOld content"
        assert strip_quoted(text) == "Sure thing."

    def test_removes_signature_block(self):
        text = "Confirming the call.\n\nBest,\nSam\n--\nSam Smith | Recruiter | +1 555 0100"
        result = strip_quoted(text)
        assert "Confirming the call." in result
        assert "555 0100" not in result

    def test_keeps_short_body_intact(self):
        text = "We would like to schedule a screening call."
        assert strip_quoted(text) == text

    def test_collapses_blank_line_runs(self):
        assert strip_quoted("A\n\n\n\n\nB") == "A\n\nB"
