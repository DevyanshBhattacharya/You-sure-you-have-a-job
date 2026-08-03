"""Surviving a flaky network.

A single dropped TLS connection used to abort an entire import and leave the
dashboard asking to start over. These tests pin the retry policy and the resume
path that replaced that behaviour.
"""

from __future__ import annotations

import socket
import ssl

import pytest
from googleapiclient.errors import HttpError

from app import backfill, statestore
from app.gmail.client import MAX_ATTEMPTS, _is_transient, with_retries


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Retry backoff is real seconds; the tests only care about the count."""
    monkeypatch.setattr("app.gmail.client.time.sleep", lambda _s: None)


class _Resp:
    def __init__(self, status):
        self.status = status
        self.reason = "test"


def http_error(status: int) -> HttpError:
    return HttpError(_Resp(status), b"{}")


class TestTransientClassification:
    @pytest.mark.parametrize(
        "exc",
        [
            # The one that actually killed the user's import: something
            # interfering with TLS mid-handshake.
            ssl.SSLError("[SSL: WRONG_VERSION_NUMBER] wrong version number"),
            socket.gaierror("name resolution failed"),
            ConnectionResetError("reset by peer"),
            TimeoutError("timed out"),
        ],
    )
    def test_network_faults_are_transient(self, exc):
        assert _is_transient(exc)

    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
    def test_rate_limits_and_server_faults_are_transient(self, status):
        assert _is_transient(http_error(status))

    @pytest.mark.parametrize("status", [400, 401, 403, 404])
    def test_real_answers_are_not_retried(self, status):
        """403 means the API is disabled. Retrying it just delays the message."""
        assert not _is_transient(http_error(status))

    def test_programming_errors_are_not_retried(self):
        assert not _is_transient(ValueError("bad argument"))


class TestRetryLoop:
    def test_recovers_after_a_transient_failure(self):
        attempts = {"n": 0}

        def call():
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise ssl.SSLError("[SSL: WRONG_VERSION_NUMBER] wrong version number")
            return "ok"

        assert with_retries(call, what="test") == "ok"
        assert attempts["n"] == 3

    def test_gives_up_after_the_limit(self):
        attempts = {"n": 0}

        def call():
            attempts["n"] += 1
            raise ssl.SSLError("still broken")

        with pytest.raises(ssl.SSLError):
            with_retries(call, what="test")
        assert attempts["n"] == MAX_ATTEMPTS

    def test_permanent_errors_fail_immediately(self):
        attempts = {"n": 0}

        def call():
            attempts["n"] += 1
            raise http_error(403)

        with pytest.raises(HttpError):
            with_retries(call, what="test")
        assert attempts["n"] == 1

    def test_success_costs_one_call(self):
        attempts = {"n": 0}

        def call():
            attempts["n"] += 1
            return 42

        assert with_retries(call, what="test") == 42
        assert attempts["n"] == 1


class TestResumeAfterInterruption:
    @pytest.fixture
    def started(self, monkeypatch):
        """Capture what resume would launch, without touching Gmail."""
        calls: list[int] = []
        monkeypatch.setattr(
            backfill, "start", lambda days, on_email=None: (calls.append(days), True)[1]
        )
        return calls

    def _set_state(self, db, status, days=None):
        statestore.set_(db, backfill.BACKFILL_STATUS, status)
        if days is not None:
            statestore.set_(db, backfill.BACKFILL_DAYS, str(days))
        db.commit()

    def test_resumes_after_an_error(self, db, started):
        self._set_state(db, "error", days=90)
        assert backfill.resume_if_interrupted() is True
        assert started == [90]

    def test_resumes_after_the_process_died_mid_fetch(self, db, started):
        # "fetching" with no live thread means the process stopped part-way.
        self._set_state(db, "fetching", days=30)
        assert backfill.resume_if_interrupted() is True
        assert started == [30]

    def test_reuses_the_original_window(self, db, started):
        """Resuming over a shorter default window would silently skip mail."""
        self._set_state(db, "error", days=365)
        backfill.resume_if_interrupted()
        assert started == [365]

    def test_completed_import_is_not_repeated(self, db, started):
        self._set_state(db, "complete", days=90)
        assert backfill.resume_if_interrupted() is False
        assert started == []

    def test_idle_state_does_nothing(self, db, started):
        self._set_state(db, "idle")
        assert backfill.resume_if_interrupted() is False
        assert started == []


class TestServiceIsNotSharedBetweenThreads:
    """`[SSL: WRONG_VERSION_NUMBER]` was a threading bug, not a TLS one.

    `build()` creates one `httplib2.Http` and every request from that service
    reuses it, but httplib2 is not thread-safe. The watcher polling while a
    backfill imports put two threads on one TLS socket; each then read back
    bytes meant for the other, and OpenSSL reported the garbled record header
    as a wrong protocol version. Retrying just made them collide again.
    """

    @pytest.fixture
    def fake_build(self, monkeypatch):
        """Count service objects handed out, without touching Google."""
        built: list[object] = []

        def build(*_a, **_kw):
            service = object()
            built.append(service)
            return service

        monkeypatch.setattr("app.gmail.auth.build", build)
        monkeypatch.setattr("app.gmail.auth.load_credentials", lambda **_kw: object())
        from app.gmail import auth

        auth.reset_service()
        return built

    def test_each_thread_gets_its_own_service(self, fake_build):
        import threading

        from app.gmail import auth

        seen: dict[int, object] = {}

        def grab() -> None:
            seen[threading.get_ident()] = auth.get_service()

        threads = [threading.Thread(target=grab) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(seen) == 4
        # The whole point: no two threads may hold the same http object.
        assert len({id(s) for s in seen.values()}) == 4

    def test_a_thread_reuses_its_own_service(self, fake_build):
        """Per-thread, not per-call — a thread keeps its connection pool."""
        from app.gmail import auth

        assert auth.get_service() is auth.get_service()
        assert len(fake_build) == 1

    def test_reset_invalidates_the_calling_thread(self, fake_build):
        from app.gmail import auth

        first = auth.get_service()
        auth.reset_service()
        assert auth.get_service() is not first

    def test_reset_invalidates_other_threads_too(self, fake_build):
        """A revoked token must not leave a stale service alive in a worker."""
        import threading

        from app.gmail import auth

        results: list[object] = []

        def grab() -> None:
            results.append(auth.get_service())

        worker = threading.Thread(target=grab)
        worker.start()
        worker.join()

        auth.reset_service()

        worker = threading.Thread(target=grab)
        worker.start()
        worker.join()

        assert results[0] is not results[1]
