# Job Mail Agent

Watches a Gmail inbox for job-related mail, keeps a structured record of every
application and its state, pushes notifications to a dashboard in real time, and
answers questions about the job search from its own knowledge base.

- **Backend** — Python, FastAPI, SQLite
- **Frontend** — React, Vite, TypeScript, Tailwind
- **Mail** — Gmail API (read-only OAuth)
- **LLM** — pluggable: Google Gemini (hosted) or Ollama (local), set by
  `LLM_PROVIDER`

---

## How it works

Nothing polls the database looking for work. Each stage publishes; the next one
consumes.

```
Gmail ──(incremental history poll)──►  Watcher
                                          │  EmailArrived
                                          ▼
                                    asyncio work queue
                                          │
                                          ▼
   Ingest ─► Prefilter ─► Classify ─► Resolve ─► Index ─► Notify
   (fetch)   (heuristic)   (Gemini)   (state    (embed)     │
                                      machine)              ▼
                                                   event bus ──► WebSocket ──► dashboard
```

`app/gmail/watcher.py` is the only module that knows *how* new mail is detected.
Swapping the poll for a Gmail Pub/Sub push webhook means adding a second
publisher that emits the same `EmailWork` item — nothing downstream changes.

A few decisions worth knowing about:

- **The prefilter is a cost lever, not a correctness one.** It only rejects mail
  it is confident about, and is biased toward false positives: letting a
  newsletter through costs one cheap model call, dropping a real interview
  invite costs an interview.
- **`Application.status` is derived and guarded by a state machine.** A late
  "thanks for applying" auto-reply cannot drag an application that already
  reached `offer` back to `applied`. The timeline (`application_events`) is
  append-only; the status is computed from it.
- **The Q&A agent has tools, not just retrieval.** "What did the recruiter say?"
  is a search question; "how many applications are in the interview stage?" is a
  SQL question. It gets both.

---

## Setup

### 1. Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Then edit `backend/.env`. `LLM_PROVIDER` picks the backend:

**Option A — Ollama (local).** No key, no quota, no bill. Slower, and the
answers are only as good as what your machine can run. This is the practical
choice while the Gemini free tier is in the way.

```powershell
winget install Ollama.Ollama      # or https://ollama.com/download
ollama serve                      # leave running in its own terminal
ollama pull qwen3:4b              # classifier + Q&A  (~2.6 GB, supports tools)
ollama pull nomic-embed-text      # embeddings        (~275 MB)
```

```ini
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3:4b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

**Option B — Gemini (hosted).** Better quality, but read the free-tier warning
below before starting a backfill.

```ini
LLM_PROVIDER=gemini
GEMINI_API_KEY=...                # https://aistudio.google.com/apikey
CLASSIFIER_MODEL=gemini-3.5-flash-lite
QA_MODEL=gemini-3.6-flash
EMBEDDING_MODEL=gemini-embedding-001
```

Other settings: `POLL_INTERVAL_SECONDS` (how often the watcher checks for new
mail) and `PROCESS_BACKLOG_ON_START` (re-queue stored-but-unclassified mail at
startup).

> **Switching providers changes the embedding dimension** (Gemini 1536 vs
> nomic-embed-text 768). Vectors of different sizes aren't comparable, so the
> store only searches ones matching the current model. Rebuild the old ones:
>
> ```powershell
> .\.venv\Scripts\python.exe scripts\reindex_kb.py
> ```

> **Verify your models before the first run.** Availability varies by key and
> tier — some models are withdrawn from new keys, and Pro models often have no
> free-tier quota at all.
>
> ```powershell
> .\.venv\Scripts\python.exe scripts\check_models.py --suggest
> ```
>
> Do **not** rely on `client.models.list()` for this. It lists models that
> return `404 … no longer available to new users` the moment you call them, and
> Pro models that return `429`. `check_models.py` probes with the same request
> the app makes, which is the only answer that means anything.

The app runs without a Gemini key — classification falls back to heuristics and
Q&A reports that it's unavailable — so you can wire up Gmail first and add the
key after.

### 2. Gmail access

Two separate steps in the Cloud console, and it's easy to do the second
without the first:

1. **Enable the Gmail API.** In the
   [Google Cloud Console](https://console.cloud.google.com/), create a project,
   then **APIs & Services → Library → Gmail API → Enable**. Creating OAuth
   credentials does *not* enable the API — skip this and every call fails with
   `403: Gmail API has not been used in project … before or it is disabled`.
2. **Create an OAuth client.** **APIs & Services → Credentials → Create
   credentials → OAuth client ID → Desktop app**. Desktop clients may redirect
   to any loopback port, so there is nothing to configure.

   > Using a **Web application** client instead? It only accepts redirect URIs
   > you registered. Add `http://localhost:8080/` (exact, trailing slash),
   > leave *Authorized JavaScript origins* empty, and keep
   > `OAUTH_REDIRECT_PORT=8080` in `.env` so the two agree.

3. Download the JSON and save it as `backend/credentials.json`.
4. Authorise once (this opens a browser):

```powershell
cd backend
.\.venv\Scripts\python.exe -m app.gmail.auth
```

That writes `backend/token.json` and prints the authorised address. Both files
are gitignored.

The scope is `gmail.readonly` — the agent never needs write access to your
mailbox.

**If something is wrong,** `GET /api/health` tells you which of the two steps
failed rather than lumping them together:

| `gmail_authorised` | `gmail_usable` | Meaning |
|---|---|---|
| `false` | `false` | No credentials — run `python -m app.gmail.auth` |
| `true` | `false` | Credentials fine, the API call failed. Read `gmail_error` / `gmail_hint` — usually the Gmail API isn't enabled, and re-authorising will not help |
| `true` | `true` | Working |

### 3. Frontend

```powershell
cd frontend
npm install
```

---

## Running

Two terminals:

```powershell
# Terminal 1 — API, watcher, agent pipeline
cd backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

```powershell
# Terminal 2 — dashboard
cd frontend
npm run dev
```

Open <http://localhost:5173>. The dev server proxies `/api` and `/ws` to the
backend, so there's no CORS or WebSocket configuration to do.

To import existing mail, click **Import mail** in the header (or
`POST /api/sync/backfill` with `{"days": 90}`). After the backfill anchors the
history cursor, the watcher takes over and new mail flows in automatically.

### Trying it without Gmail

```powershell
cd backend
.\.venv\Scripts\python.exe scripts\seed_demo.py          # populate sample data
.\.venv\Scripts\python.exe scripts\seed_demo.py --clear  # remove it again
```

Everything it creates carries a `demo-` id prefix, so `--clear` removes exactly
what it added.

---

## Iterating on the classifier

Prompt changes are cheap to evaluate — replay against mail you already have,
without re-hitting Gmail:

```powershell
cd backend
.\.venv\Scripts\python.exe scripts\replay.py --dry-run       # show diffs, write nothing
.\.venv\Scripts\python.exe scripts\replay.py --limit 50      # apply
.\.venv\Scripts\python.exe scripts\replay.py --only-job-related
```

Manual corrections made in the **Inbox** tab are stored with
`classification_source = "manual"`, which makes them a ready-made evaluation set.

---

## Cost and the free-tier ceiling

Token usage is accumulated per call and reported on `/api/health`
(`llm_calls`, `prompt_tokens`, `output_tokens`), so a runaway backfill shows up
immediately rather than on the bill. The prefilter drops obvious non-job mail
before any model call.

> ⚠️ **The Gemini free tier is measured per day, and it is small.** Observed
> limits: `gemini-3.6-flash` **20 requests/day**, `gemini-3.5-flash-lite`
> **15/day**, `gemini-2.0-flash` **0** (no free quota). That is nowhere near
> enough to classify a real inbox — a few hundred emails would take weeks.
>
> Two ways out: **enable billing** on the Cloud project (Flash-Lite is cheap
> enough that a few hundred emails costs a few cents), or run
> **`LLM_PROVIDER=ollama`** locally, where there is no quota at all.

The pipeline is built for this constraint rather than defeated by it:

- A quota error (`429`) or an unreachable backend (Ollama not running)
  **never** produces a fabricated verdict. The email is left with
  `processed_at = NULL` and retried on the next start — degrading to a guess
  would freeze a low-confidence result derived from the sender's name, and
  nothing re-examines processed mail. Both conditions are systemic, so a
  fallback would corrupt the entire backlog rather than one message.
- Once quota is exhausted the pipeline **stops calling the API** for the stated
  retry window (an hour if the limit is per-day) instead of hammering it.
  `/api/sync/status` exposes `quota_blocked`, `quota_retry_in_seconds` and
  `quota_deferred`.
- The startup sweep queues **job-signal mail first**, so a small daily
  allowance is spent on likely interview invites rather than on newsletters
  that happen to be newer.

`GET /api/health` reports `emails_unprocessed` so a stalled backlog is visible.

---

## Tests

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check app
```

```powershell
cd frontend
npm run build   # type-checks and builds
```

The suite covers MIME parsing and quoted-reply stripping, the prefilter's golden
set, every state-machine transition (including the ones that must be refused),
application matching and merging, pipeline idempotency, vector search, the HTTP
surface, and the realtime bus → WebSocket path. None of it needs Gmail
credentials or an API key.

---

## Layout

```
backend/
  app/
    gmail/      auth, API client, MIME normalisation, watcher
    agent/
      providers/  base (the seam), gemini, ollama
      llm.py      facade over the selected provider
      prefilter, classify, resolve, pipeline, tools, qa
    kb/         chunking, embeddings, vector store, indexer
    api/        HTTP + WebSocket routes
    events.py   work queue + event bus
    models.py   ORM and domain enums
  scripts/      check_models.py, replay.py, reindex_kb.py, seed_demo.py
  tests/
frontend/
  src/
    api/        typed client, SSE reader
    hooks/      WebSocket subscription
    pages/      Dashboard, Applications, Inbox, Ask
    components/ shared UI
```

---

## Notes

- `credentials.json`, `token.json`, `.env` and `*.db` are gitignored. Check
  before committing.
- The vector store is brute-force cosine over an in-memory matrix — sub-millisecond
  for a personal mailbox, and no native SQLite extension to install. `VectorStore`
  in `app/kb/store.py` is the seam if the corpus ever outgrows that.
- Gmail expires history cursors after roughly a week. The watcher detects this
  and falls back to a windowed sweep, so leaving the app off for a while is safe.
