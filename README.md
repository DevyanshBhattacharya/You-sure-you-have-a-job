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

- **The prefilter turns on a strong/weak signal split.** A *strong* signal —
  "your application", an interview, an assessment, or mail from an employer's
  own recruiting address — overrides every bulk-mail rule, because a genuine ATS
  acknowledgement carries `List-Unsubscribe` and usually lands in Promotions. A
  *weak* one (the word "job" somewhere, or "linkedin" in the sender) overrides
  nothing. Getting that backwards is what filled the board with digests: on a
  real mailbox, 96% of mail was rejected without a model call once being a job
  board stopped counting as a signal, and no genuine application was lost.
- **The board tracks applications you made, not jobs that exist.** Being job
  *related* and being *yours* are separate questions, so the classifier answers
  both (`is_job_related`, `recipient_applied`) and only the second opens an
  application. Job alerts, adverts and cold outreach stay searchable in the
  inbox and the knowledge base; they just never reach the board.
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
OLLAMA_NUM_CTX=16384
OLLAMA_EXTRACTION_NUM_CTX=6144
```

> ⚠️ **`OLLAMA_NUM_CTX` is not a tuning knob — leave it large.** Ollama gives
> every model a **4096-token** window by default, whatever the model actually
> supports (qwen3 supports 262k). Past that it silently discards the *oldest*
> tokens, which are the system instruction and the question. There is no error
> and no warning: the model answers a prompt it can no longer read.
>
> A full email, or a tool result listing every application, clears 4096 easily.
> Observed on this project at the default: asked "how many applications am I
> tracking?", the model returned a 400-word report about "a simulated dataset"
> and never gave a number. At 16384 the same question answers correctly.
>
> **But a window is not free, which is why there are two of them.** Its KV cache
> is allocated in VRAM beside the weights, so an oversized one pushes layers onto
> the CPU. Measured here on qwen3:4b against a 4 GB card (GTX 1650), footprint
> from `ollama ps`:
>
> | Window | Model footprint | Fits in 4 GB? |
> |---|---|---|
> | 16384 | 5.09 GB | no — only 2.65 GB stayed resident, half the layers on CPU |
> | 6144 | 3.52 GB | yes |
>
> So `OLLAMA_NUM_CTX` sizes *chat*, where tool results really are large, and
> `OLLAMA_EXTRACTION_NUM_CTX` sizes classification, whose prompt is capped by
> `classify.MAX_PROMPT_BODY_CHARS`. Truncation is still reported for both, so
> raise the extraction window if you ever see that warning.
>
> Note that Ollama reloads the model whenever the requested window changes, so
> asking a question mid-import costs one reload each way. Classification runs in
> bulk, so this is one reload per switch, not per email.
>
> **If classification is slow, check memory before tuning anything.** On a 4 GB
> card a 4B model is already marginal; if system RAM is also short the runner
> pages and a *ten-token* prompt can fail to return in 90 seconds. `nvidia-smi`
> and free RAM tell you this in one look. The cheapest fix is a smaller
> classifier — extraction is a much easier job than chat, and the two models are
> already separate settings:
>
> ```ini
> OLLAMA_MODEL=qwen3:1.7b        # classification: high volume, simple task
> OLLAMA_QA_MODEL=qwen3:4b       # chat: low volume, needs the reasoning
> ```
>
> Confirm what a loaded model is really using — this is the ground truth, not
> the model's advertised capacity:
>
> ```powershell
> ollama ps      # the CONTEXT column, and how much sits in VRAM
> ```
>
> The adapter logs a warning when Ollama reports evaluating a prompt right up
> to the ceiling, which is the visible symptom of an invisible truncation.

> **Leave `OLLAMA_THINK=true` for reasoning models** (qwen3, deepseek-r1).
> Setting it false does *not* stop them reasoning — it stops Ollama separating
> the reasoning into its own field, so it arrives in `content` and gets shown
> as the answer. Verified on Ollama 0.32.5 with qwen3:4b.

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

**You never have to import by hand.** On first start the backend imports on its
own (`AUTO_BACKFILL_ON_START`), then the watcher polls every
`POLL_INTERVAL_SECONDS` and new mail arrives without being asked. The header
button stays as a manual override and for changing the window
(`POST /api/sync/backfill` with `{"days": 90}`). Specifically:

- Mail already stored is skipped on any later import, matched on `gmail_id`, so
  re-running costs nothing and never duplicates.
- Transient network faults (dropped TLS, DNS, `429`, `5xx`) are retried with
  backoff instead of aborting the run. A single `[SSL: WRONG_VERSION_NUMBER]`
  used to kill an entire import.
- An import that still ends early — crash, laptop closed, network gone — is
  **resumed automatically on the next start** and picks up where it stopped.
  Set `RESUME_BACKFILL_ON_START=false` to require a manual click instead.
- Each message is handed to the agent as it arrives, so an interrupted import
  keeps the classification work it already did.
- Imported-but-unclassified mail is re-swept every `BACKLOG_SWEEP_SECONDS`, so
  a remainder past `BACKLOG_BATCH_LIMIT` — or work lost from the in-memory queue
  when the process stopped — drains without a restart.

If the watcher is off for more than about a week Gmail expires the history
cursor; it detects that and falls back to a windowed sweep **sized to the actual
gap** since `last_sync_at`, not a fixed window. A machine off for a month would
otherwise come back, sweep three days, reset the cursor, and lose the rest
permanently.

> **Importing and classifying are different stages, at very different speeds.**
> A local model spends tens of seconds per email, so an import can report
> complete while the board is still empty — which looks exactly like nothing
> happened, and invites pressing Import again (which correctly does nothing,
> since every message is already fetched). The header therefore reports
> whichever stage is actually busy: `Importing 12/13`, then `Classifying · 96
> left`, then `Watching for new mail`.

> **`[SSL: WRONG_VERSION_NUMBER]` from the Gmail client is a threading bug, not
> a TLS one.** `googleapiclient.discovery.build()` creates a single
> `httplib2.Http`, every request from that service reuses it, and httplib2 is
> not thread-safe. With the watcher polling while an import runs, two threads
> write to one TLS socket and each reads back the other's bytes; OpenSSL reports
> the garbled record header as a wrong protocol version. It looks like a proxy
> or a firewall and is neither, and retrying only makes the threads collide
> again. `app/gmail/auth.py` therefore hands out **one service per thread**.
> Measured over 12 concurrent calls: shared 6 failures, per-thread 0.

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
`classification_source = "manual"`, which makes them a ready-made evaluation
set — and they **act**, rather than just relabelling:

- "Not job related" retracts the application the old verdict created (unless
  other emails still support it).
- "Job related" re-runs extraction with the prefilter skipped, so the message
  reaches the board even if a rule rejected it.

The manual verdict is sticky. A correction that lasted only until the next
replay would just be re-rejected by the rule the person was overriding.

A replay that flips an email from job-related to not **retracts** what the old
verdict recorded: its timeline entry and notifications go, and an application
left with no events at all is deleted. Without that, a run of bad verdicts could
never be cleaned up — the corrected emails would drop out while the applications
they invented stayed on the board.

So recovering from a bad classifier run is just a replay:

```powershell
.\.venv\Scripts\python.exe scripts\replay.py --dry-run   # see what would change
.\.venv\Scripts\python.exe scripts\replay.py             # apply
```

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

## Deploying this anywhere but your laptop

Read this before exposing the port. The app serves the full text of one
mailbox and will summarise it on request, so the default posture — no
authentication — is safe on loopback and nowhere else.

**1. Set a token.** `APP_AUTH_TOKEN` guards every `/api` and `/ws` route.

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Unset, the server runs open and logs a warning at startup saying so. Set, it
also disables `/docs`, `/redoc` and `/openapi.json`. The dashboard asks for the
token on its first 401 and keeps it in `localStorage` — deliberately not a
cookie, because a cookie is attached to cross-site requests automatically and
this app has no CSRF defence.

**2. Terminate TLS in front of it.** A bearer token over plain HTTP is readable
by anything on the path, and the WebSocket passes it in the query string
(browsers cannot set headers on a handshake), where proxies log it.

**3. Name your origins.** `CORS_ORIGINS` must list exact origins, never `*`.
It also vets the `Origin` on WebSocket handshakes — CORS does not apply to
WebSockets, so without that check any page you visit could open a socket to a
localhost server and read your notification feed.

**4. Bind deliberately.** `--host 127.0.0.1` unless something in front of it is
doing the TLS and the token is set.

### What this is not

**Single-user by construction.** There is one Gmail token on disk, one
database, one shared secret. Nothing in the data model is scoped to a user —
`GET /api/applications` means *the* applications, not *yours*. Serving a second
person means per-user OAuth, a tenant column on every table, and authorisation
on every query. That is a rewrite of the data layer, not a login screen; do not
mistake the token for a step toward it.

**Untrusted input reaches the model, by design.** Anyone can email you, and that
mail is fed to the classifier and quoted back by the Q&A agent. Both system
prompts state that message text is data and never instruction. Constrained
decoding bounds the damage — the classifier can only emit the extraction schema,
and every tool is read-only — so the worst case is a wrong verdict on one email,
which the Inbox override corrects. It is a real limitation, not a solved
problem.

**No rate limiting.** `POST /api/chat` triggers model work on every call. Behind
the token that is your own usage; if you ever drop the token, put a limiter in
front.

---

## Notes

- `credentials.json`, `token.json`, `.env` and `*.db` are gitignored. Check
  before committing.
- The vector store is brute-force cosine over an in-memory matrix — sub-millisecond
  for a personal mailbox, and no native SQLite extension to install. `VectorStore`
  in `app/kb/store.py` is the seam if the corpus ever outgrows that.
- Gmail expires history cursors after roughly a week. The watcher detects this
  and falls back to a windowed sweep, so leaving the app off for a while is safe.
