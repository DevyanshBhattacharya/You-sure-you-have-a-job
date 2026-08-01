# Job Mail Agent

Watches a Gmail inbox for job-related mail, keeps a structured record of every
application and its state, pushes notifications to a dashboard in real time, and
answers questions about the job search from its own knowledge base.

- **Backend** — Python, FastAPI, SQLite
- **Frontend** — React, Vite, TypeScript, Tailwind
- **Mail** — Gmail API (read-only OAuth)
- **LLM** — Google Gemini (`google-genai`)

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

Then edit `backend/.env`:

| Variable | What it's for |
|---|---|
| `GEMINI_API_KEY` | Get one at <https://aistudio.google.com/apikey> |
| `CLASSIFIER_MODEL` | Per-email extraction. A fast/cheap model is the right choice here. |
| `QA_MODEL` | Dashboard Q&A. Worth a stronger model. |
| `EMBEDDING_MODEL` | Knowledge-base embeddings |
| `POLL_INTERVAL_SECONDS` | How often the watcher checks for new mail |

> **Model IDs move.** The defaults were current when this was written. Check
> what your key can actually reach:
>
> ```powershell
> .\.venv\Scripts\python.exe -c "from google import genai; [print(m.name) for m in genai.Client().models.list()]"
> ```

The app runs without a Gemini key — classification falls back to heuristics and
Q&A reports that it's unavailable — so you can wire up Gmail first and add the
key after.

### 2. Gmail access

1. In the [Google Cloud Console](https://console.cloud.google.com/), create a
   project and enable the **Gmail API**.
2. **APIs & Services → Credentials → Create credentials → OAuth client ID →
   Desktop app**.
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

## Cost

Token usage is accumulated per call and reported on `/api/health`
(`llm_calls`, `prompt_tokens`, `output_tokens`), so a runaway backfill shows up
immediately rather than on the bill. The prefilter drops obvious non-job mail
before any model call.

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
    agent/      llm, prefilter, classify, resolve, pipeline, tools, qa
    kb/         chunking, embeddings, vector store, indexer
    api/        HTTP + WebSocket routes
    events.py   work queue + event bus
    models.py   ORM and domain enums
  scripts/      replay.py, seed_demo.py
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
