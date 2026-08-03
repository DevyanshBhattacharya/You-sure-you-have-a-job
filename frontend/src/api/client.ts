import type {
  Application,
  ApplicationDetail,
  ApplicationStatus,
  BoardResponse,
  ChatEvent,
  DashboardStats,
  EmailDetail,
  EmailSummary,
  Health,
  // Aliased: `Notification` is also a DOM global, so an unaliased import
  // would be shadowed silently and the return type would be wrong.
  Notification as AppNotification,
  NotificationPage,
  Page,
  SyncStatus,
} from "./types";

export class ApiError extends Error {
  // Declared explicitly rather than as a constructor parameter property —
  // this project builds with `erasableSyntaxOnly`, which disallows those.
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/**
 * Shared secret for a protected deployment.
 *
 * Kept in localStorage rather than a cookie on purpose: a cookie would be
 * attached automatically to cross-site requests, which is exactly the CSRF
 * shape this app has no defence against. A header has to be set deliberately,
 * so a hostile page cannot borrow the session.
 */
const TOKEN_KEY = "jobmail.token";

export function getToken(): string {
  return localStorage.getItem(TOKEN_KEY) ?? "";
}

export function setToken(token: string): void {
  if (token) localStorage.setItem(TOKEN_KEY, token);
  else localStorage.removeItem(TOKEN_KEY);
}

function authHeaders(): Record<string, string> {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...authHeaders(), ...init?.headers },
  });

  if (!response.ok) {
    // FastAPI puts the useful part in `detail`; fall back to the status text.
    let detail = response.statusText;
    try {
      const body = await response.json();
      if (typeof body?.detail === "string") detail = body.detail;
      else if (Array.isArray(body?.detail)) detail = body.detail[0]?.msg ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(detail, response.status);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

function query(params: Record<string, string | number | boolean | undefined>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== "") search.set(key, String(value));
  }
  const qs = search.toString();
  return qs ? `?${qs}` : "";
}

export const api = {
  health: () => request<Health>("/api/health"),

  syncStatus: () => request<SyncStatus>("/api/sync/status"),
  startBackfill: (days?: number) =>
    request<SyncStatus>("/api/sync/backfill", {
      method: "POST",
      body: JSON.stringify({ days: days ?? null }),
    }),

  emails: (params: {
    job_related?: boolean;
    unclassified?: boolean;
    q?: string;
    limit?: number;
    offset?: number;
  }) => request<Page<EmailSummary>>(`/api/emails${query(params)}`),

  email: (id: number) => request<EmailDetail>(`/api/emails/${id}`),

  setClassification: (id: number, isJobRelated: boolean) =>
    request<EmailDetail>(`/api/emails/${id}/classification`, {
      method: "POST",
      body: JSON.stringify({ is_job_related: isJobRelated }),
    }),

  board: () => request<BoardResponse>("/api/applications/board"),
  stats: () => request<DashboardStats>("/api/applications/stats"),

  applications: (params: { status?: string; company?: string; limit?: number }) =>
    request<Page<Application>>(`/api/applications${query(params)}`),

  application: (id: number) => request<ApplicationDetail>(`/api/applications/${id}`),

  setStatus: (id: number, status: ApplicationStatus) =>
    request<Application>(`/api/applications/${id}/status`, {
      method: "POST",
      body: JSON.stringify({ status }),
    }),

  setNotes: (id: number, notes: string) =>
    request<Application>(`/api/applications/${id}/notes`, {
      method: "POST",
      body: JSON.stringify({ notes }),
    }),

  notifications: (unreadOnly = false) =>
    request<NotificationPage>(`/api/notifications${query({ unread_only: unreadOnly })}`),

  markRead: (id: number) =>
    request<AppNotification>(`/api/notifications/${id}/read`, { method: "POST" }),

  markAllRead: () => request<NotificationPage>("/api/notifications/read-all", { method: "POST" }),
};

/**
 * Stream a chat answer. Uses fetch + a manual SSE reader rather than
 * EventSource, because EventSource cannot issue a POST.
 */
export async function streamChat(
  message: string,
  history: { role: string; content: string }[],
  onEvent: (event: ChatEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ message, history }),
    signal,
  });

  if (!response.ok || !response.body) {
    throw new ApiError(`Chat request failed (${response.status})`, response.status);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line; a partial frame stays buffered.
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";

    for (const frame of frames) {
      const line = frame.split("\n").find((l) => l.startsWith("data:"));
      if (!line) continue;
      try {
        onEvent(JSON.parse(line.slice(5).trim()) as ChatEvent);
      } catch {
        /* ignore malformed frame */
      }
    }
  }
}
