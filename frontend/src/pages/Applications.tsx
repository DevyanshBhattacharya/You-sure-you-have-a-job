import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "../api/client";
import type { ApplicationStatus } from "../api/types";
import {
  Card,
  EmptyState,
  ErrorNote,
  Spinner,
  StatusBadge,
  STATUS_LABELS,
  formatDate,
  formatDateTime,
  relativeTime,
} from "../components/ui";

const ALL_STATUSES = Object.keys(STATUS_LABELS) as ApplicationStatus[];

function TimelineDrawer({ id, onClose }: { id: number; onClose: () => void }) {
  const queryClient = useQueryClient();
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["application", id],
    queryFn: () => api.application(id),
  });

  const setStatus = useMutation({
    mutationFn: (status: ApplicationStatus) => api.setStatus(id, status),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["application", id] });
      queryClient.invalidateQueries({ queryKey: ["applications"] });
      queryClient.invalidateQueries({ queryKey: ["board"] });
      queryClient.invalidateQueries({ queryKey: ["stats"] });
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

      <aside className="relative flex h-full w-full max-w-lg flex-col overflow-y-auto border-l border-slate-200 bg-white shadow-xl dark:border-slate-800 dark:bg-slate-900">
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
            <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/95 px-6 py-4 backdrop-blur dark:border-slate-800 dark:bg-slate-900/95">
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0">
                  <h2 className="truncate text-lg font-semibold text-slate-900 dark:text-slate-50">
                    {data.company}
                  </h2>
                  {data.role_title && (
                    <p className="truncate text-sm text-slate-600 dark:text-slate-400">
                      {data.role_title}
                    </p>
                  )}
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
                <StatusBadge status={data.status} />
                <select
                  value={data.status}
                  onChange={(e) => setStatus.mutate(e.target.value as ApplicationStatus)}
                  className="rounded-md border border-slate-300 bg-white px-2 py-1 text-xs text-slate-700 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-300"
                >
                  {ALL_STATUSES.map((status) => (
                    <option key={status} value={status}>
                      Set to {STATUS_LABELS[status]}
                    </option>
                  ))}
                </select>
              </div>
            </header>

            <div className="space-y-6 px-6 py-5">
              <dl className="grid grid-cols-2 gap-x-4 gap-y-3 text-sm">
                {[
                  ["Location", data.location],
                  ["Source", data.source],
                  ["Contact", data.contact_email],
                  ["Salary", data.salary_text],
                  ["First seen", formatDate(data.first_seen_at)],
                  ["Last activity", formatDate(data.last_activity_at)],
                ].map(([label, value]) => (
                  <div key={label as string}>
                    <dt className="text-xs text-slate-500 dark:text-slate-400">{label}</dt>
                    <dd className="truncate text-slate-800 dark:text-slate-200">
                      {value || "—"}
                    </dd>
                  </div>
                ))}
              </dl>

              {data.next_action && (
                <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 dark:border-amber-900 dark:bg-amber-950/40">
                  <p className="text-xs font-semibold tracking-wide text-amber-900 uppercase dark:text-amber-300">
                    Next action
                  </p>
                  <p className="mt-1 text-sm text-amber-900 dark:text-amber-200">
                    {data.next_action}
                  </p>
                  {data.next_action_due && (
                    <p className="mt-1 text-xs text-amber-700 dark:text-amber-400">
                      Due {formatDate(data.next_action_due)}
                    </p>
                  )}
                </div>
              )}

              {data.job_url && (
                <a
                  href={data.job_url}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-block text-sm text-sky-600 underline underline-offset-2 dark:text-sky-400"
                >
                  Job posting
                </a>
              )}

              <section>
                <h3 className="mb-3 text-xs font-semibold tracking-wide text-slate-500 uppercase dark:text-slate-400">
                  Timeline
                </h3>
                <ol className="relative space-y-4 border-l border-slate-200 pl-5 dark:border-slate-800">
                  {data.timeline.map((entry) => (
                    <li key={entry.id} className="relative">
                      <span className="absolute top-1.5 -left-[1.44rem] h-2 w-2 rounded-full bg-slate-400 dark:bg-slate-600" />
                      <p className="text-sm font-medium text-slate-800 dark:text-slate-200">
                        {entry.event_type.replace(/_/g, " ")}
                        {entry.status_before !== entry.status_after && (
                          <span className="ml-2 text-xs font-normal text-slate-500 dark:text-slate-400">
                            {entry.status_before} → {entry.status_after}
                          </span>
                        )}
                      </p>
                      {entry.summary && (
                        <p className="mt-0.5 text-sm text-slate-600 dark:text-slate-400">
                          {entry.summary}
                        </p>
                      )}
                      <p className="mt-0.5 text-xs text-slate-400 dark:text-slate-500">
                        {formatDateTime(entry.occurred_at)}
                      </p>
                    </li>
                  ))}
                  {data.timeline.length === 0 && (
                    <li className="text-sm text-slate-500">No events recorded.</li>
                  )}
                </ol>
              </section>
            </div>
          </>
        )}
      </aside>
    </div>
  );
}

export default function Applications({
  openId,
  onOpen,
}: {
  openId: number | null;
  onOpen: (id: number | null) => void;
}) {
  const [status, setStatus] = useState<string>("");
  const [company, setCompany] = useState("");

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["applications", status, company],
    queryFn: () => api.applications({ status: status || undefined, company: company || undefined }),
  });

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <input
          value={company}
          onChange={(e) => setCompany(e.target.value)}
          placeholder="Filter by company…"
          className="w-56 rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 placeholder:text-slate-400 focus:ring-2 focus:ring-sky-500 focus:outline-none dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
        />
        <select
          value={status}
          onChange={(e) => setStatus(e.target.value)}
          className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
        >
          <option value="">All statuses</option>
          {ALL_STATUSES.map((s) => (
            <option key={s} value={s}>
              {STATUS_LABELS[s]}
            </option>
          ))}
        </select>
        {data && (
          <span className="text-sm text-slate-500 dark:text-slate-400">
            {data.total} application{data.total === 1 ? "" : "s"}
          </span>
        )}
      </div>

      {isError && <ErrorNote error={error} />}
      {isLoading && <Spinner />}

      {data && data.items.length === 0 && (
        <EmptyState title="Nothing matches those filters" />
      )}

      {data && data.items.length > 0 && (
        <Card className="overflow-hidden">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-slate-200 bg-slate-50 text-xs tracking-wide text-slate-500 uppercase dark:border-slate-800 dark:bg-slate-950/50 dark:text-slate-400">
              <tr>
                <th className="px-4 py-2.5 font-medium">Company</th>
                <th className="px-4 py-2.5 font-medium">Role</th>
                <th className="px-4 py-2.5 font-medium">Status</th>
                <th className="px-4 py-2.5 font-medium">Next action</th>
                <th className="px-4 py-2.5 font-medium">Activity</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
              {data.items.map((application) => (
                <tr
                  key={application.id}
                  onClick={() => onOpen(application.id)}
                  className="cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-800/50"
                >
                  <td className="px-4 py-3 font-medium text-slate-900 dark:text-slate-100">
                    {application.company}
                  </td>
                  <td className="px-4 py-3 text-slate-600 dark:text-slate-400">
                    {application.role_title ?? "—"}
                  </td>
                  <td className="px-4 py-3">
                    <StatusBadge status={application.status} />
                  </td>
                  <td className="max-w-xs truncate px-4 py-3 text-slate-600 dark:text-slate-400">
                    {application.next_action ?? "—"}
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-slate-500 dark:text-slate-500">
                    {relativeTime(application.last_activity_at)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}

      {openId !== null && <TimelineDrawer id={openId} onClose={() => onOpen(null)} />}
    </div>
  );
}
