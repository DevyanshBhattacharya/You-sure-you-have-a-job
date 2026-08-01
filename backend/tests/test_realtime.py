"""The realtime path: agent finishes an email -> browser hears about it.

This is the chain the whole design rests on, so it gets tested directly rather
than inferred from the pieces.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.agent import pipeline
from app.events import (
    APPLICATION_UPDATED,
    EMAIL_PROCESSED,
    NOTIFICATION_CREATED,
    EmailWork,
    bus,
    work_queue,
)
from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def read_until(ws, topic: str, limit: int = 10) -> dict:
    """Read frames until `topic` arrives, skipping keepalive pings."""
    for _ in range(limit):
        frame = ws.receive_json()
        if frame["topic"] == topic:
            return frame
    raise AssertionError(f"never received a {topic!r} frame")


class TestBusToSocket:
    def test_a_published_notification_reaches_the_browser(self, client):
        with client.websocket_connect("/ws/notifications") as ws:
            assert ws.receive_json()["topic"] == "connected"

            # publish_threadsafe is the path the watcher and backfill threads
            # actually use, so exercise that rather than a direct publish.
            bus.publish_threadsafe(
                NOTIFICATION_CREATED,
                {"id": 1, "title": "Offer from Acme", "priority": "high"},
            )

            frame = read_until(ws, NOTIFICATION_CREATED)
            assert frame["data"]["title"] == "Offer from Acme"
            assert frame["data"]["priority"] == "high"

    def test_application_updates_are_broadcast(self, client):
        with client.websocket_connect("/ws/notifications") as ws:
            ws.receive_json()
            bus.publish_threadsafe(APPLICATION_UPDATED, {"id": 7, "status": "interviewing"})

            frame = read_until(ws, APPLICATION_UPDATED)
            assert frame["data"]["status"] == "interviewing"

    def test_every_connected_client_receives_the_same_event(self, client):
        with (
            client.websocket_connect("/ws/notifications") as first,
            client.websocket_connect("/ws/notifications") as second,
        ):
            first.receive_json()
            second.receive_json()

            bus.publish_threadsafe(EMAIL_PROCESSED, {"id": 42})

            assert read_until(first, EMAIL_PROCESSED)["data"]["id"] == 42
            assert read_until(second, EMAIL_PROCESSED)["data"]["id"] == 42

    def test_unsubscribed_after_disconnect(self, client):
        with client.websocket_connect("/ws/notifications") as ws:
            ws.receive_json()
        # The subscription is torn down on close; publishing must not raise.
        bus.publish_threadsafe(EMAIL_PROCESSED, {"id": 1})


class TestPipelinePublishing:
    def test_publish_emits_one_frame_per_change(self, monkeypatch):
        published: list[tuple[str, dict]] = []
        monkeypatch.setattr(
            bus, "publish", lambda topic, payload: published.append((topic, payload))
        )

        pipeline._publish(
            {
                "email": {"id": 1, "is_job_related": True},
                "application": {"id": 2, "company": "Acme"},
                "notifications": [{"id": 3, "title": "Offer from Acme"}],
            }
        )

        assert [topic for topic, _ in published] == [
            EMAIL_PROCESSED,
            APPLICATION_UPDATED,
            NOTIFICATION_CREATED,
        ]

    def test_non_job_mail_only_emits_the_email_frame(self, monkeypatch):
        published: list[str] = []
        monkeypatch.setattr(bus, "publish", lambda topic, _payload: published.append(topic))

        pipeline._publish(
            {"email": {"id": 1, "is_job_related": False}, "application": None, "notifications": []}
        )

        assert published == [EMAIL_PROCESSED]


class TestWorkQueue:
    async def test_submitted_work_is_delivered_in_order(self):
        work_queue.reset()
        await work_queue.submit(EmailWork(gmail_id="a"))
        await work_queue.submit(EmailWork(email_id=7))

        first = await work_queue.get()
        second = await work_queue.get()

        assert first.gmail_id == "a"
        assert second.email_id == 7

    def test_submitting_without_a_loop_drops_rather_than_raises(self, monkeypatch):
        """A watcher tick during shutdown must not take the thread down."""
        monkeypatch.setattr("app.events.get_loop", lambda: None)
        work_queue.submit_threadsafe(EmailWork(gmail_id="x"))  # no exception
