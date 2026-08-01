import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "../api/client";
import { Card, EmptyState, ErrorNote, Spinner, formatDateTime } from "../components/ui";

type Filter = "job" | "not-job" | "unclassified" | "all";

const FILTERS: { key: Filter; label: string }[] = [
  { key: "job", label: "Job related" },
  { key: "not-job", label: "Filtered out" },
  { key: "unclassified", label: "Unclassified" },
  { key: "all", label: "All" },
];

function paramsFor(filter: Filter, q: string) {
  const base = { q: q || undefined, limit: 100 };
  switch (filter) {
    case "job":
      return { ...base, job_related: true };
    case "not-job":
      return { ...base, job_related: false };
    case "unclassified":
      return { ...base, unclassified: true };
    default:
      return base;
  }
}

function Confidence({ value, source }: { value: number | null; source: string | null }) {
  if (value === null) return <span className="text-xs text-slate-400">—</span>;
  const pct = Math.round(value * 100);
  const tone =
    pct >= 85
      ? "text-emerald-600 dark:text-emerald-400"
      : pct >= 60
        ? "text-amber-600 dark:text-amber-400"
        : "text-rose-600 dark:text-rose-400";
  return (
    <span className={`text-xs tabular-nums ${tone}`}>
      {pct}%
      {source && <span className="ml-1 text-slate-400 dark:text-slate-600">{source}</span>}
    </span>
  );
}

function EmailDrawer({ id, onClose }: { id: number; onClose: () => void }) {
  const queryClient = useQueryClient();
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["email", id],
    queryFn: () => api.email(id),
  });

  const override = useMutation({
    mutationFn: (isJobRelated: boolean) => api.setClassification(id, isJobRelated),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["email", id] });
      queryClient.invalidateQueries({ queryKey: ["emails"] });
    },
  });

  return (
    <div className="fixed inset-0 z-40 flex justify-end">
      <button
        type="button"
        aria-label="Close"
        onClick={onClose}
        className="absolute inset-0 bg-slate-900/40 backdrop-blur-[1px]"
      />

      <aside className="relative flex h-full w-full max-w-2xl flex-col overflow-y-auto border-l border-slate-200 bg-white shadow-xl dark:border-slate-800 dark:bg-slate-900">
        {isLoading && (
          <div className="p-6">
            <Spinner />
          </div>
        )}
        {isError && (
          <div className="p-6">
            <ErrorNote error={error} />
          </div>
        )}

        {data && (
          <>
            <header className="sticky top-0 border-b border-slate-200 bg-white/95 px-6 py-4 backdrop-blur dark:border-slate-800 dark:bg-slate-900/95">
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0">
                  <h2 className="text-base font-semibold text-slate-900 dark:text-slate-50">
                    {data.subject ?? "(no subject)"}
                  </h2>
                  <p className="mt-0.5 truncate text-sm text-slate-500 dark:text-slate-400">
                    {data.from_name ? `${data.from_name} · ` : ""}
                    {data.from_addr}
                  </p>
                  <p className="text-xs text-slate-400 dark:text-slate-500">
                    {formatDateTime(data.received_at)}
                  </p>
                </div>
                <button
                  type="button"
                  onClick={onClose}
                  className="rounded-md px-2 py-1 text-sm text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800"
                >
                  Close
                </button>
              </div>

              <div className="mt-3 flex items-center gap-2">
                <span className="text-xs text-slate-500 dark:text-slate-400">
                  Classified as
                </span>
                <button
                  type="button"
                  disabled={override.isPending}
                  onClick={() => override.mutate(true)}
                  className={`rounded-md px-2.5 py-1 text-xs font-medium transition ${
                    data.is_job_related
                      ? "bg-emerald-600 text-white"
                      : "bg-slate-100 text-slate-600 hover:bg-slate-200 dark:bg-slate-800 dark:text-slate-400"
                  }`}
                >
                  Job related
                </button>
                <button
                  type="button"
                  disabled={override.isPending}
                  onClick={() => override.mutate(false)}
                  className={`rounded-md px-2.5 py-1 text-xs font-medium transition ${
                    data.is_job_related === false
                      ? "bg-slate-700 text-white dark:bg-slate-600"
                      : "bg-slate-100 text-slate-600 hover:bg-slate-200 dark:bg-slate-800 dark:text-slate-400"
                  }`}
                >
                  Not job related
                </button>
                <Confidence
                  value={data.classification_confidence}
                  source={data.classification_source}
                />
              </div>
            </header>

            <div className="px-6 py-5">
              <pre className="font-sans text-sm leading-relaxed whitespace-pre-wrap text-slate-700 dark:text-slate-300">
                {data.body_text ?? data.snippet ?? "(empty message)"}
              </pre>

              {data.classification_raw && (
                <details className="mt-6">
                  <summary className="cursor-pointer text-xs font-medium text-slate-500 dark:text-slate-400">
                    Extracted fields
                  </summary>
                  <pre className="mt-2 overflow-x-auto rounded-lg bg-slate-50 p-3 font-mono text-xs text-slate-700 dark:bg-slate-950 dark:text-slate-300">
                    {JSON.stringify(data.classification_raw, null, 2)}
                  </pre>
                </details>
              )}
            </div>
          </>
        )}
      </aside>
    </div>
  );
}

export default function Inbox() {
  const [filter, setFilter] = useState<Filter>("job");
  const [q, setQ] = useState("");
  const [openId, setOpenId] = useState<number | null>(null);

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["emails", filter, q],
    queryFn: () => api.emails(paramsFor(filter, q)),
  });

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex rounded-lg border border-slate-300 p-0.5 dark:border-slate-700">
          {FILTERS.map((f) => (
            <button
              key={f.key}
              type="button"
              onClick={() => setFilter(f.key)}
              className={`rounded-md px-3 py-1.5 text-sm font-medium transition ${
                filter === f.key
                  ? "bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900"
                  : "text-slate-600 hover:bg-slate-100 dark:text-slate-400 dark:hover:bg-slate-800"
              }`}
            >
              {f.label}
            </button>
          ))}
        </div>

        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search subject or sender…"
          className="w-64 rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 placeholder:text-slate-400 focus:ring-2 focus:ring-sky-500 focus:outline-none dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
        />

        {data && (
          <span className="text-sm text-slate-500 dark:text-slate-400">{data.total} messages</span>
        )}
      </div>

      {isError && <ErrorNote error={error} />}
      {isLoading && <Spinner />}

      {data && data.items.length === 0 && (
        <EmptyState
          title="No messages here"
          hint="Import mail with a backfill, or switch filters."
        />
      )}

      {data && data.items.length > 0 && (
        <Card className="divide-y divide-slate-100 dark:divide-slate-800">
          {data.items.map((email) => (
            <button
              key={email.id}
              type="button"
              onClick={() => setOpenId(email.id)}
              className="flex w-full items-start gap-4 px-4 py-3 text-left hover:bg-slate-50 dark:hover:bg-slate-800/50"
            >
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium text-slate-900 dark:text-slate-100">
                  {email.subject ?? "(no subject)"}
                </p>
                <p className="truncate text-xs text-slate-500 dark:text-slate-400">
                  {email.from_name || email.from_addr}
                </p>
                {email.snippet && (
                  <p className="mt-0.5 truncate text-xs text-slate-400 dark:text-slate-500">
                    {email.snippet}
                  </p>
                )}
              </div>
              <div className="flex shrink-0 flex-col items-end gap-1">
                <span className="text-xs whitespace-nowrap text-slate-400 dark:text-slate-500">
                  {formatDateTime(email.received_at)}
                </span>
                <Confidence
                  value={email.classification_confidence}
                  source={email.classification_source}
                />
              </div>
            </button>
          ))}
        </Card>
      )}

      {openId !== null && <EmailDrawer id={openId} onClose={() => setOpenId(null)} />}
    </div>
  );
}
