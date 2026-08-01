"""Chunking and vector search."""

from __future__ import annotations

import numpy as np

from app.kb.chunk import TARGET_CHARS, chunk_text
from app.kb.store import NumpyVectorStore, from_blob, hydrate_hits, to_blob
from app.models import KBChunk
from tests.fixtures import make_email


class TestChunking:
    def test_short_text_is_one_chunk(self):
        chunks = chunk_text("A short email body.")
        assert len(chunks) == 1
        assert chunks[0].text == "A short email body."

    def test_empty_text_yields_nothing(self):
        assert chunk_text("") == []
        assert chunk_text("   \n\n  ") == []

    def test_header_is_repeated_on_every_chunk(self):
        body = "\n\n".join(["Paragraph " + "x" * 500 for _ in range(12)])
        chunks = chunk_text(body, header="[2026-03-02] From: Acme | Subject: Offer")
        assert len(chunks) > 1
        assert all(c.text.startswith("[2026-03-02] From: Acme") for c in chunks)

    def test_long_text_splits_into_multiple_chunks(self):
        body = "\n\n".join([f"Paragraph {i}. " + "word " * 200 for i in range(10)])
        chunks = chunk_text(body)
        assert len(chunks) > 1
        assert all(c.text for c in chunks)

    def test_a_single_oversized_paragraph_is_broken_up(self):
        body = "word " * (TARGET_CHARS // 2)
        chunks = chunk_text(body)
        assert len(chunks) > 1

    def test_chunk_indices_are_sequential(self):
        body = "\n\n".join([f"Para {i}. " + "word " * 200 for i in range(8)])
        chunks = chunk_text(body)
        assert [c.index for c in chunks] == list(range(len(chunks)))


class TestBlobRoundTrip:
    def test_vector_survives_serialisation(self):
        vector = [0.1, -0.25, 0.75, 1.0]
        restored = from_blob(to_blob(vector))
        assert np.allclose(restored, np.array(vector, dtype=np.float32))


def _add_chunk(db, email, text: str, vector: list[float], application_id=None) -> KBChunk:
    chunk = KBChunk(
        email_id=email.id,
        application_id=application_id,
        chunk_index=0,
        text=text,
        embedding=to_blob(vector),
        dim=len(vector),
        token_count=len(text) // 4,
    )
    db.add(chunk)
    db.flush()
    return chunk


class TestVectorSearch:
    def test_returns_the_nearest_vector_first(self, db):
        email = make_email(db)
        _add_chunk(db, email, "about cats", [1.0, 0.0, 0.0])
        _add_chunk(db, email, "about dogs", [0.0, 1.0, 0.0])
        _add_chunk(db, email, "about birds", [0.0, 0.0, 1.0])
        db.commit()

        store = NumpyVectorStore()
        hits = store.search(db, [0.9, 0.1, 0.0], k=3)

        assert hits[0].text == "about cats"
        assert hits[0].score > hits[1].score

    def test_magnitude_does_not_affect_ranking(self, db):
        """Cosine similarity, so a longer vector in the same direction ranks
        the same as a short one."""
        email = make_email(db)
        _add_chunk(db, email, "target", [10.0, 0.0, 0.0])
        _add_chunk(db, email, "other", [0.0, 1.0, 0.0])
        db.commit()

        store = NumpyVectorStore()
        assert store.search(db, [0.01, 0.0, 0.0], k=2)[0].text == "target"

    def test_k_limits_results(self, db):
        email = make_email(db)
        for i in range(6):
            _add_chunk(db, email, f"chunk {i}", [float(i), 1.0, 0.0])
        db.commit()

        assert len(NumpyVectorStore().search(db, [1.0, 1.0, 0.0], k=2)) == 2

    def test_filters_by_application(self, db):
        from app.models import Application

        app_a = Application(company="Acme", company_normalized="acme")
        app_b = Application(company="Globex", company_normalized="globex")
        db.add_all([app_a, app_b])
        db.flush()

        email = make_email(db)
        _add_chunk(db, email, "acme text", [1.0, 0.0, 0.0], application_id=app_a.id)
        _add_chunk(db, email, "globex text", [1.0, 0.0, 0.0], application_id=app_b.id)
        db.commit()

        hits = NumpyVectorStore().search(db, [1.0, 0.0, 0.0], k=5, application_id=app_b.id)
        assert [h.text for h in hits] == ["globex text"]

    def test_empty_index_returns_nothing(self, db):
        assert NumpyVectorStore().search(db, [1.0, 0.0, 0.0], k=5) == []

    def test_zero_query_vector_returns_nothing(self, db):
        email = make_email(db)
        _add_chunk(db, email, "something", [1.0, 0.0, 0.0])
        db.commit()

        assert NumpyVectorStore().search(db, [0.0, 0.0, 0.0], k=5) == []

    def test_dimension_mismatch_is_handled_not_raised(self, db):
        """Changing EMBEDDING_DIM mid-corpus must degrade, not crash."""
        email = make_email(db)
        _add_chunk(db, email, "three dims", [1.0, 0.0, 0.0])
        db.commit()

        assert NumpyVectorStore().search(db, [1.0, 0.0], k=5) == []

    def test_cache_refreshes_when_chunks_are_added(self, db):
        email = make_email(db)
        _add_chunk(db, email, "first", [1.0, 0.0, 0.0])
        db.commit()

        store = NumpyVectorStore()
        assert len(store.search(db, [1.0, 0.0, 0.0], k=5)) == 1

        _add_chunk(db, email, "second", [0.9, 0.1, 0.0])
        db.commit()

        assert len(store.search(db, [1.0, 0.0, 0.0], k=5)) == 2


class TestHydrateHits:
    def test_attaches_email_and_application_context(self, db):
        from app.models import Application

        application = Application(company="Acme", company_normalized="acme", role_title="SRE")
        db.add(application)
        db.flush()

        email = make_email(db, subject="Interview invitation")
        _add_chunk(db, email, "some text", [1.0, 0.0, 0.0], application_id=application.id)
        db.commit()

        hits = NumpyVectorStore().search(db, [1.0, 0.0, 0.0], k=1)
        rows = hydrate_hits(db, hits)

        assert rows[0]["subject"] == "Interview invitation"
        assert rows[0]["company"] == "Acme"
        assert rows[0]["role_title"] == "SRE"
        assert rows[0]["email_id"] == email.id
