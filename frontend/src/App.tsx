import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "./api/client";
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

function SyncButton() {
  const queryClient = useQueryClient();
  const { data: sync } = useQuery({
    queryKey: ["sync"],
    queryFn: api.syncStatus,
    // Only poll while an import is actually in flight.
    refetchInterval: (query) => (query.state.data?.running ? 2000 : false),
  });

  const start = useMutation({
    mutationFn: () => api.startBackfill(),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["sync"] }),
  });

  if (sync?.running) {
    const pct = sync.total > 0 ? Math.round((sync.done / sync.total) * 100) : 0;
    return (
      <span className="text-xs text-slate-500 tabular-nums dark:text-slate-400">
        Importing {sync.done}/{sync.total} ({pct}%)
      </span>
    );
  }

  return (
    <div className="flex items-center gap-2">
      {start.isError && (
        <span
          title={(start.error as Error).message}
          className="max-w-56 truncate text-xs text-rose-600 dark:text-rose-400"
        >
          {(start.error as Error).message}
        </span>
      )}
      <button
        type="button"
        onClick={() => start.mutate()}
        disabled={start.isPending}
        className="rounded-lg border border-slate-300 px-3 py-1.5 text-xs font-medium text-slate-700 transition hover:bg-slate-100 disabled:opacity-50 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800"
      >
        {start.isPending ? "Starting…" : "Import mail"}
      </button>
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState<Tab>("dashboard");
  const [openApplication, setOpenApplication] = useState<number | null>(null);

  // Single WebSocket for the whole app — mounted once, here.
  const { state, toasts, dismissToast } = useNotifications();

  const { data: health } = useQuery({ queryKey: ["health"], queryFn: api.health });
  const { data: notifications } = useQuery({
    queryKey: ["notifications"],
    queryFn: () => api.notifications(false),
  });

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
            {health && !health.gmail_authorised && (
              <span
                title="Run `python -m app.gmail.auth` in the backend directory"
                className="rounded-lg bg-amber-100 px-2.5 py-1 text-xs font-medium text-amber-900 dark:bg-amber-950 dark:text-amber-300"
              >
                Gmail not connected
              </span>
            )}
            {health?.gmail_address && (
              <span className="hidden text-xs text-slate-500 sm:inline dark:text-slate-400">
                {health.gmail_address}
              </span>
            )}
            <SyncButton />
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
