import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import type { Application, BoardColumn } from "../api/types";
import {
  Card,
  EmptyState,
  ErrorNote,
  PriorityDot,
  Spinner,
  StatTile,
  STATUS_LABELS,
  formatDate,
  relativeTime,
} from "../components/ui";

function ApplicationCard({
  application,
  onOpen,
}: {
  application: Application;
  onOpen: (id: number) => void;
}) {
  const overdue =
    application.next_action_due && new Date(application.next_action_due) < new Date();

  return (
    <button
      type="button"
      onClick={() => onOpen(application.id)}
      className="w-full rounded-lg border border-slate-200 bg-white p-3 text-left transition hover:border-slate-300 hover:shadow-sm focus:ring-2 focus:ring-sky-500 focus:outline-none dark:border-slate-800 dark:bg-slate-900 dark:hover:border-slate-700"
    >
      <p className="truncate text-sm font-semibold text-slate-900 dark:text-slate-100">
        {application.company}
      </p>
      {application.role_title && (
        <p className="mt-0.5 truncate text-xs text-slate-600 dark:text-slate-400">
          {application.role_title}
        </p>
      )}

      {application.next_action && (
        <p
          className={`mt-2 line-clamp-2 text-xs ${
            overdue
              ? "font-medium text-rose-600 dark:text-rose-400"
              : "text-slate-500 dark:text-slate-400"
          }`}
        >
          {application.next_action}
          {application.next_action_due && ` · ${formatDate(application.next_action_due)}`}
        </p>
      )}

      <p className="mt-2 text-[11px] text-slate-400 dark:text-slate-500">
        {relativeTime(application.last_activity_at)}
      </p>
    </button>
  );
}

function Column({
  column,
  onOpen,
}: {
  column: BoardColumn;
  onOpen: (id: number) => void;
}) {
  return (
    <div className="flex w-64 shrink-0 flex-col">
      <div className="mb-2 flex items-center justify-between px-1">
        <h3 className="text-xs font-semibold tracking-wide text-slate-600 uppercase dark:text-slate-400">
          {STATUS_LABELS[column.status] ?? column.status}
        </h3>
        <span className="rounded-full bg-slate-200 px-2 text-xs font-medium text-slate-700 tabular-nums dark:bg-slate-800 dark:text-slate-300">
          {column.count}
        </span>
      </div>

      <div className="flex max-h-[calc(100vh-22rem)] flex-col gap-2 overflow-y-auto pr-1">
        {column.items.length === 0 ? (
          <p className="rounded-lg border border-dashed border-slate-200 px-3 py-6 text-center text-xs text-slate-400 dark:border-slate-800 dark:text-slate-600">
            Empty
          </p>
        ) : (
          column.items.map((application) => (
            <ApplicationCard key={application.id} application={application} onOpen={onOpen} />
          ))
        )}
      </div>
    </div>
  );
}

function ActivityFeed() {
  const { data } = useQuery({
    queryKey: ["notifications"],
    queryFn: () => api.notifications(false),
  });

  const items = data?.items.slice(0, 12) ?? [];

  return (
    <Card className="flex h-full flex-col">
      <div className="border-b border-slate-200 px-4 py-3 dark:border-slate-800">
        <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
          Live activity
        </h2>
        <p className="text-xs text-slate-500 dark:text-slate-400">
          Pushed by the agent as mail arrives
        </p>
      </div>

      <div className="flex-1 divide-y divide-slate-100 overflow-y-auto dark:divide-slate-800">
        {items.length === 0 ? (
          <p className="px-4 py-8 text-center text-xs text-slate-400 dark:text-slate-600">
            Nothing yet.
          </p>
        ) : (
          items.map((notification) => (
            <div key={notification.id} className="flex gap-2.5 px-4 py-3">
              <PriorityDot priority={notification.priority} />
              <div className="min-w-0 flex-1">
                <p className="text-sm leading-snug font-medium text-slate-800 dark:text-slate-200">
                  {notification.title}
                </p>
                {notification.body && (
                  <p className="mt-0.5 line-clamp-2 text-xs text-slate-500 dark:text-slate-400">
                    {notification.body}
                  </p>
                )}
                <p className="mt-1 text-[11px] text-slate-400 dark:text-slate-500">
                  {relativeTime(notification.created_at)}
                </p>
              </div>
            </div>
          ))
        )}
      </div>
    </Card>
  );
}

export default function Dashboard({ onOpen }: { onOpen: (id: number) => void }) {
  const stats = useQuery({ queryKey: ["stats"], queryFn: api.stats });
  const board = useQuery({ queryKey: ["board"], queryFn: api.board });

  if (stats.isError) return <ErrorNote error={stats.error} />;
  if (board.isError) return <ErrorNote error={board.error} />;
  if (stats.isLoading || board.isLoading) return <Spinner />;

  const s = stats.data;
  const columns = board.data?.columns ?? [];
  const isEmpty = columns.every((c) => c.count === 0);

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        <StatTile label="Active" value={s?.active ?? 0} hint="in play" />
        <StatTile label="Interviewing" value={s?.interviewing ?? 0} hint="incl. screens" />
        <StatTile label="Offers" value={s?.offers ?? 0} accent />
        <StatTile label="Due in 7d" value={s?.actions_due_7d ?? 0} hint="actions" />
        <StatTile label="Stale" value={s?.awaiting_reply ?? 0} hint="14d+ silent" />
        <StatTile label="Rejected" value={s?.rejected ?? 0} />
      </div>

      {isEmpty ? (
        <EmptyState
          title="No applications tracked yet"
          hint="Run a backfill from the header to import your recent mail, or wait for the watcher to pick up the next message."
        />
      ) : (
        <div className="grid gap-6 xl:grid-cols-[1fr_20rem]">
          <Card className="overflow-hidden p-4">
            <div className="flex gap-4 overflow-x-auto pb-2">
              {columns.map((column) => (
                <Column key={column.status} column={column} onOpen={onOpen} />
              ))}
            </div>
          </Card>

          <div className="min-h-80 xl:h-[calc(100vh-18rem)]">
            <ActivityFeed />
          </div>
        </div>
      )}
    </div>
  );
}
