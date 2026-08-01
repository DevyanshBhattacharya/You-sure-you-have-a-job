import type { ReactNode } from "react";
import type { ApplicationStatus, Priority } from "../api/types";

export const STATUS_LABELS: Record<ApplicationStatus, string> = {
  discovered: "Discovered",
  applied: "Applied",
  acknowledged: "Acknowledged",
  screening: "Screening",
  assessment: "Assessment",
  interviewing: "Interviewing",
  offer: "Offer",
  accepted: "Accepted",
  rejected: "Rejected",
  ghosted: "Ghosted",
  withdrawn: "Withdrawn",
};

const STATUS_CLASSES: Record<ApplicationStatus, string> = {
  discovered: "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
  applied: "bg-sky-100 text-sky-800 dark:bg-sky-950 dark:text-sky-300",
  acknowledged: "bg-cyan-100 text-cyan-800 dark:bg-cyan-950 dark:text-cyan-300",
  screening: "bg-violet-100 text-violet-800 dark:bg-violet-950 dark:text-violet-300",
  assessment: "bg-amber-100 text-amber-900 dark:bg-amber-950 dark:text-amber-300",
  interviewing: "bg-indigo-100 text-indigo-800 dark:bg-indigo-950 dark:text-indigo-300",
  offer: "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300",
  accepted: "bg-emerald-600 text-white dark:bg-emerald-700",
  rejected: "bg-rose-100 text-rose-800 dark:bg-rose-950 dark:text-rose-300",
  ghosted: "bg-stone-100 text-stone-700 dark:bg-stone-800 dark:text-stone-300",
  withdrawn: "bg-stone-100 text-stone-700 dark:bg-stone-800 dark:text-stone-300",
};

export function StatusBadge({ status }: { status: ApplicationStatus }) {
  return (
    <span
      className={`inline-flex shrink-0 items-center rounded-full px-2 py-0.5 text-xs font-medium ${
        STATUS_CLASSES[status] ?? STATUS_CLASSES.discovered
      }`}
    >
      {STATUS_LABELS[status] ?? status}
    </span>
  );
}

export function PriorityDot({ priority }: { priority: Priority }) {
  const colour =
    priority === "high" ? "bg-rose-500" : priority === "normal" ? "bg-sky-500" : "bg-slate-400";
  return <span className={`mt-1.5 h-2 w-2 shrink-0 rounded-full ${colour}`} />;
}

export function Card({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`rounded-xl border border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-900 ${className}`}
    >
      {children}
    </div>
  );
}

export function StatTile({
  label,
  value,
  hint,
  accent = false,
}: {
  label: string;
  value: number | string;
  hint?: string;
  accent?: boolean;
}) {
  return (
    <Card className="p-4">
      <p className="text-xs font-medium tracking-wide text-slate-500 uppercase dark:text-slate-400">
        {label}
      </p>
      <p
        className={`mt-1 text-3xl font-semibold tabular-nums ${
          accent ? "text-emerald-600 dark:text-emerald-400" : "text-slate-900 dark:text-slate-50"
        }`}
      >
        {value}
      </p>
      {hint && <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">{hint}</p>}
    </Card>
  );
}

export function EmptyState({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-slate-300 px-6 py-12 text-center dark:border-slate-700">
      <p className="text-sm font-medium text-slate-600 dark:text-slate-300">{title}</p>
      {hint && <p className="mt-1 max-w-sm text-xs text-slate-500 dark:text-slate-400">{hint}</p>}
    </div>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-slate-500 dark:text-slate-400">
      <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-slate-300 border-t-slate-600 dark:border-slate-700 dark:border-t-slate-300" />
      {label ?? "Loading…"}
    </div>
  );
}

export function ErrorNote({ error }: { error: unknown }) {
  const message = error instanceof Error ? error.message : String(error);
  return (
    <div className="rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800 dark:border-rose-900 dark:bg-rose-950/50 dark:text-rose-300">
      {message}
    </div>
  );
}

/** Compact relative time, falling back to an absolute date past a week. */
export function relativeTime(iso: string | null): string {
  if (!iso) return "—";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "—";

  const seconds = Math.round((Date.now() - then) / 1000);
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  if (seconds < 604800) return `${Math.floor(seconds / 86400)}d ago`;
  return new Date(iso).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

export function formatDate(iso: string | null): string {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

export function formatDateTime(iso: string | null): string {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}
