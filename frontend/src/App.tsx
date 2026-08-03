import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { ApiError, api, setToken } from "./api/client";
import { PriorityDot } from "./components/ui";
import { useNotifications } from "./hooks/useNotifications";
import Applications from "./pages/Applications";
import Ask from "./pages/Ask";
import Dashboard from "./pages/Dashboard";
import Inbox from "./pages/Inbox";

type Tab = "dashboard" | "applications" | "inbox" | "ask";

const TABS: { key: Tab; label: string }[] = [
  { key: "dashboard", label: "Dashboard" },
  { key: "applications", label: "Applications" },
  { key: "inbox", label: "Inbox" },
  { key: "ask", label: "Ask" },
];

function ConnectionPill({ state }: { state: "connecting" | "open" | "closed" }) {
  const config = {
    open: { dot: "bg-emerald-500", label: "Live" },
    connecting: { dot: "bg-amber-500 animate-pulse", label: "Connecting" },
    closed: { dot: "bg-rose-500", label: "Offline" },
  }[state];

  return (
    <span
      title="Realtime connection to the agent"
      className="inline-flex items-center gap-1.5 rounded-full border border-slate-200 px-2.5 py-1 text-xs text-slate-600 dark:border-slate-800 dark:text-slate-400"
    >
      <span className={`h-1.5 w-1.5 rounded-full ${config.dot}`} />
      {config.label}
    </span>
  );
}

/**
 * Import and classification state.
 *
 * Deliberately not just a button. Importing and classifying are separate
 * stages: a local model spends tens of seconds on each email, so the import can
 * report "complete" while the board stays empty for a long time afterwards.
 * Showing only import progress made that look like nothing had happened, and
 * the only affordance on screen was to press Import again — which correctly did
 * nothing, because every message had already been fetched.
 *
 * So the widget reports whichever stage is actually busy, surfaces a failed or
 * interrupted import with the reason, and keeps the manual trigger as a
 * fallback rather than the main path. The server imports on its own.
 */
function SyncIndicator() {
  const queryClient = useQueryClient();
  const { data: sync } = useQuery({
    queryKey: ["sync"],
    queryFn: api.syncStatus,
    // Poll fast while something is moving, slowly otherwise — the server can
    // start an import by itself, so "idle" still has to be observed.
    refetchInterval: (query) => {
      const s = query.state.data;
      if (!s) return 5_000;
      return s.running || s.pending_classification > 0 ? 2_000 : 20_000;
    },
  });

  const start = useMutation({
    mutationFn: () => api.startBackfill(),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["sync"] }),
  });

  // Refresh the board when the last email finishes classifying. The WebSocket
  // normally does this; this covers the case where it dropped mid-run.
  const wasPending = useRef(0);
  useEffect(() => {
    const pending = sync?.pending_classification ?? 0;
    if (wasPending.current > 0 && pending === 0) {
      for (const key of ["board", "stats", "applications", "emails", "notifications"]) {
        queryClient.invalidateQueries({ queryKey: [key] });
      }
    }
    wasPending.current = pending;
  }, [sync?.pending_classification, queryClient]);

  const failed = sync?.status === "error" || sync?.status === "interrupted";
  const retry = (
    <button
      type="button"
      onClick={() => start.mutate()}
      disabled={start.isPending || sync?.running}
      className="rounded-lg border border-slate-300 px-3 py-1.5 text-xs font-medium text-slate-700 transition hover:bg-slate-100 disabled:opacity-50 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800"
    >
      {start.isPending ? "Starting…" : failed ? "Retry import" : "Import mail"}
    </button>
  );

  if (sync?.running) {
    const pct = sync.total > 0 ? Math.round((sync.done / sync.total) * 100) : 0;
    return (
      <span className="text-xs text-slate-500 tabular-nums dark:text-slate-400">
        Importing {sync.done}/{sync.total} ({pct}%)
      </span>
    );
  }

  if (sync?.quota_blocked) {
    return (
      <div className="flex items-center gap-2">
        <span
          title={sync.quota_reason}
          className="rounded-lg bg-amber-100 px-2.5 py-1 text-xs font-medium text-amber-900 dark:bg-amber-950 dark:text-amber-300"
        >
          Classifier paused · retrying in {sync.quota_retry_in_seconds}s
        </span>
        {retry}
      </div>
    );
  }

  if ((sync?.pending_classification ?? 0) > 0) {
    return (
      <span className="text-xs text-slate-500 tabular-nums dark:text-slate-400">
        Classifying · {sync?.pending_classification} left
      </span>
    );
  }

  return (
    <div className="flex items-center gap-2">
      {failed && (
        <span
          title={sync?.error ?? undefined}
          className="max-w-64 truncate text-xs text-rose-600 dark:text-rose-400"
        >
          Import {sync?.status}: {sync?.error ?? "unknown error"}
        </span>
      )}
      {!failed && start.isError && (
        <span
          title={(start.error as Error).message}
          className="max-w-56 truncate text-xs text-rose-600 dark:text-rose-400"
        >
          {(start.error as Error).message}
        </span>
      )}
      {!failed && sync?.watcher_running && (
        <span
          title={
            sync.last_sync_at
              ? `Last checked ${new Date(sync.last_sync_at).toLocaleString()}`
              : undefined
          }
          className="hidden text-xs text-slate-500 lg:inline dark:text-slate-400"
        >
          Watching for new mail
        </span>
      )}
      {retry}
    </div>
  );
}

/**
 * Token prompt for a protected deployment.
 *
 * Shown only when the server actually rejects a request with 401, so a local
 * run with no `APP_AUTH_TOKEN` never sees it. Not a login — there are no users
 * here, just one shared secret guarding one person's mailbox.
 */
function TokenGate({ onSaved }: { onSaved: () => void }) {
  const [value, setValue] = useState("");

  return (
    <div className="flex min-h-screen items-center justify-center bg-slate-50 px-6 dark:bg-slate-950">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (!value.trim()) return;
          setToken(value.trim());
          onSaved();
        }}
        className="w-full max-w-sm rounded-xl border border-slate-200 bg-white p-6 dark:border-slate-800 dark:bg-slate-900"
      >
        <h1 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
          Access token required
        </h1>
        <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
          This server has <code className="font-mono">APP_AUTH_TOKEN</code> set. Paste it to
          continue; it is stored in this browser only.
        </p>
        <input
          type="password"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          autoFocus
          className="mt-4 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 focus:ring-2 focus:ring-sky-500 focus:outline-none dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
        />
        <button
          type="submit"
          disabled={!value.trim()}
          className="mt-3 w-full rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-40 dark:bg-slate-100 dark:text-slate-900"
        >
          Continue
        </button>
      </form>
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState<Tab>("dashboard");
  const [openApplication, setOpenApplication] = useState<number | null>(null);

  // Single WebSocket for the whole app — mounted once, here.
  const { state, toasts, dismissToast } = useNotifications();

  const health = useQuery({ queryKey: ["health"], queryFn: api.health });
  const { data: notifications } = useQuery({
    queryKey: ["notifications"],
    queryFn: () => api.notifications(false),
  });

  // The server is the authority on whether a token is needed; asking for one up
  // front would put a password box in front of every local dev run.
  const queryClient = useQueryClient();
  const unauthorised = health.error instanceof ApiError && health.error.status === 401;
  if (unauthorised) {
    return <TokenGate onSaved={() => queryClient.invalidateQueries()} />;
  }

  const openApplicationTab = (id: number) => {
    setOpenApplication(id);
    setTab("applications");
  };

  return (
    <div className="min-h-full bg-slate-50 text-slate-900 dark:bg-slate-950 dark:text-slate-100">
      <header className="sticky top-0 z-30 border-b border-slate-200 bg-white/90 backdrop-blur dark:border-slate-800 dark:bg-slate-950/90">
        <div className="mx-auto flex max-w-[92rem] flex-wrap items-center gap-x-6 gap-y-3 px-6 py-3">
          <div className="flex items-center gap-2.5">
            <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-slate-900 text-xs font-bold text-white dark:bg-slate-100 dark:text-slate-900">
              JA
            </span>
            <h1 className="text-sm font-semibold">Job Mail Agent</h1>
          </div>

          <nav className="flex gap-1">
            {TABS.map((t) => (
              <button
                key={t.key}
                type="button"
                onClick={() => setTab(t.key)}
                className={`relative rounded-lg px-3 py-1.5 text-sm font-medium transition ${
                  tab === t.key
                    ? "bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900"
                    : "text-slate-600 hover:bg-slate-100 dark:text-slate-400 dark:hover:bg-slate-800"
                }`}
              >
                {t.label}
                {t.key === "dashboard" && (notifications?.unread ?? 0) > 0 && (
                  <span className="absolute -top-0.5 -right-0.5 flex h-4 min-w-4 items-center justify-center rounded-full bg-rose-500 px-1 text-[10px] font-semibold text-white">
                    {notifications?.unread}
                  </span>
                )}
              </button>
            ))}
          </nav>

          <div className="ml-auto flex items-center gap-3">
            {health.data && !health.data.gmail_usable && (
              <span
                title={[health.data.gmail_error, health.data.gmail_hint].filter(Boolean).join(" ")}
                className="max-w-96 truncate rounded-lg bg-amber-100 px-2.5 py-1 text-xs font-medium text-amber-900 dark:bg-amber-950 dark:text-amber-300"
              >
                {/* Distinguish "no credentials" from "credentials fine, API
                    unreachable" — the fixes are completely different. */}
                {health.data.gmail_authorised
                  ? (health.data.gmail_error ?? "Gmail unavailable")
                  : "Gmail not connected"}
              </span>
            )}
            {health.data?.gmail_address && (
              <span className="hidden text-xs text-slate-500 sm:inline dark:text-slate-400">
                {health.data.gmail_address}
              </span>
            )}
            <SyncIndicator />
            <ConnectionPill state={state} />
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[92rem] px-6 py-6">
        {tab === "dashboard" && <Dashboard onOpen={openApplicationTab} />}
        {tab === "applications" && (
          <Applications openId={openApplication} onOpen={setOpenApplication} />
        )}
        {tab === "inbox" && <Inbox />}
        {tab === "ask" && <Ask />}
      </main>

      {toasts.length > 0 && (
        <div className="pointer-events-none fixed right-4 bottom-4 z-50 flex w-80 flex-col gap-2">
          {toasts.map((toast) => (
            <div
              key={toast.id}
              className="pointer-events-auto flex gap-2.5 rounded-xl border border-slate-200 bg-white p-3 shadow-lg dark:border-slate-700 dark:bg-slate-900"
            >
              <PriorityDot priority={toast.priority} />
              <div className="min-w-0 flex-1">
                <p className="text-sm leading-snug font-medium text-slate-900 dark:text-slate-100">
                  {toast.title}
                </p>
                {toast.body && (
                  <p className="mt-0.5 line-clamp-3 text-xs text-slate-500 dark:text-slate-400">
                    {toast.body}
                  </p>
                )}
              </div>
              <button
                type="button"
                onClick={() => dismissToast(toast.id)}
                className="shrink-0 self-start text-xs text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
