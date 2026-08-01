export type ApplicationStatus =
  | "discovered"
  | "applied"
  | "acknowledged"
  | "screening"
  | "assessment"
  | "interviewing"
  | "offer"
  | "accepted"
  | "rejected"
  | "ghosted"
  | "withdrawn";

export type Priority = "high" | "normal" | "low";

export interface Health {
  status: string;
  /** We hold credentials. */
  gmail_authorised: boolean;
  /** We can actually call the API — false if e.g. the API isn't enabled. */
  gmail_usable: boolean;
  gmail_address: string | null;
  gmail_error: string | null;
  gmail_hint: string | null;
  emails_stored: number;
  applications: number;
  watcher_running: boolean;
  llm_calls: number;
  prompt_tokens: number;
  output_tokens: number;
}

export interface SyncStatus {
  running: boolean;
  status: string | null;
  total: number;
  done: number;
  error: string | null;
  last_history_id: string | null;
}

export interface EmailSummary {
  id: number;
  gmail_id: string;
  thread_id: string | null;
  from_addr: string | null;
  from_name: string | null;
  subject: string | null;
  snippet: string | null;
  received_at: string | null;
  is_job_related: boolean | null;
  classification_confidence: number | null;
  classification_source: string | null;
}

export interface EmailDetail extends EmailSummary {
  to_addr: string | null;
  body_text: string | null;
  labels: string[] | null;
  classification_raw: Record<string, unknown> | null;
  processed_at: string | null;
}

export interface Page<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}

export interface Application {
  id: number;
  company: string;
  role_title: string | null;
  location: string | null;
  source: string | null;
  status: ApplicationStatus;
  job_url: string | null;
  salary_text: string | null;
  contact_name: string | null;
  contact_email: string | null;
  next_action: string | null;
  next_action_due: string | null;
  first_seen_at: string;
  last_activity_at: string;
}

export interface TimelineEntry {
  id: number;
  email_id: number | null;
  event_type: string;
  occurred_at: string;
  summary: string | null;
  confidence: number | null;
  status_before: string | null;
  status_after: string | null;
}

export interface ApplicationDetail extends Application {
  notes: string | null;
  timeline: TimelineEntry[];
}

export interface BoardColumn {
  status: ApplicationStatus;
  count: number;
  items: Application[];
}

export interface BoardResponse {
  columns: BoardColumn[];
}

export interface DashboardStats {
  total: number;
  active: number;
  interviewing: number;
  offers: number;
  rejected: number;
  awaiting_reply: number;
  actions_due_7d: number;
}

export interface Notification {
  id: number;
  kind: string;
  priority: Priority;
  title: string;
  body: string | null;
  application_id: number | null;
  email_id: number | null;
  created_at: string;
  read_at: string | null;
}

export interface NotificationPage {
  items: Notification[];
  total: number;
  unread: number;
}

export interface Citation {
  email_id: number | null;
  application_id: number | null;
  subject: string | null;
  company: string | null;
  received_at: string | null;
}

/** Frames pushed over the dashboard WebSocket. */
export type SocketMessage =
  | { topic: "connected"; data: { topics: string[] } }
  | { topic: "ping"; data: Record<string, never> }
  | { topic: "notification.created"; data: Notification }
  | { topic: "application.updated"; data: Record<string, unknown> }
  | { topic: "email.processed"; data: Record<string, unknown> }
  | { topic: string; data: Record<string, unknown> };

/** Events on the chat SSE stream. */
export type ChatEvent =
  | { type: "tool"; name: string; args: Record<string, unknown> }
  | { type: "token"; text: string }
  | { type: "done"; citations: Citation[]; tool_calls: string[] }
  | { type: "error"; message: string };
