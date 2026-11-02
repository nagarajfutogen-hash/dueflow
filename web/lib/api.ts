// Every call goes to FastAPI on the same origin with the API token the owner pasted in
// (kept in sessionStorage, like the /dev pages). No secrets are built into the bundle.

const TOKEN_KEY = "dueflow-token";

export function getToken(): string {
  try {
    return sessionStorage.getItem(TOKEN_KEY) ?? "";
  } catch {
    return "";
  }
}

export function setToken(token: string): void {
  try {
    sessionStorage.setItem(TOKEN_KEY, token);
  } catch {
    /* private mode: the token lives only in this page */
  }
}

export function clearToken(): void {
  try {
    sessionStorage.removeItem(TOKEN_KEY);
  } catch {
    /* nothing stored */
  }
}

// Fired when the API says the token is missing or wrong; the layout sends the user to login.
export const LOGOUT_EVENT = "dueflow-logout";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    // Line-by-line errors when the API sends a list (for example a CSV import).
    public details: string[] = [],
  ) {
    super(message);
  }
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${getToken()}`,
      ...(init.headers ?? {}),
    },
  });
  const body = await res.json().catch(() => null);
  if (res.status === 401 && !path.startsWith("/auth/")) {
    clearToken();
    window.dispatchEvent(new Event(LOGOUT_EVENT));
  }
  if (!res.ok) {
    const detail = body && typeof body.detail === "string" ? body.detail : res.statusText;
    const details =
      body && Array.isArray(body.detail) && body.detail.every((d: unknown) => typeof d === "string")
        ? (body.detail as string[])
        : [];
    throw new ApiError(res.status, `${res.status}: ${detail}`, details);
  }
  return body as T;
}

export type Failure = {
  eval_run_id: number;
  scenario_id: string;
  run_index: number;
  conversation_id: number | null;
  suite: string;
  agent_version: string;
  label: string;
  run_status: string;
  simulator_version: string;
  started_at: string;
  categories: string[];
  promoted: boolean;
};

export type Turn = {
  seq: number;
  speaker: "agent" | "debtor";
  text: string;
  source: "script" | "llm" | null;
  prompt_version: string | null;
  latency_ms: Record<string, number | boolean> | null;
};

export type AuditEvent = {
  id: number;
  type: string;
  rule: string | null;
  decision: string | null;
  reason: string | null;
  input: Record<string, unknown>;
  clock_time: string | null;
};

export type Score = {
  eval_run_id: number;
  run_index: number;
  metric: string;
  value: number;
  passed: boolean | null;
  evidence: Record<string, unknown>;
};

export type Detail = {
  conversation: {
    id: number;
    channel: string;
    label: string;
    agent_version: string;
    scenario_id: string | null;
    outcome: string | null;
    sim_started_at: string;
    llm_calls: number;
    prompt_tokens: number;
    completion_tokens: number;
  };
  turns: Turn[];
  events: AuditEvent[];
  scores: Score[];
  categories: string[];
};

export type Estimate = {
  scenario_id: string;
  n: number;
  per_side: number | null;
  total: number | null;
};

export type ReplayRow = {
  id: number;
  scenario_id: string;
  version_a: string;
  version_b: string;
  status: "running" | "completed" | "failed";
  run_a_id: number | null;
  run_b_id: number | null;
  estimate: Partial<Estimate>;
  report: string | null;
  error: string | null;
  created_at: string;
  finished_at: string | null;
  label: string;
  k: number;
};

// --- Admin dashboard (PLANNING §16) ---------------------------------------------------------

export type Period = "today" | "7d" | "30d" | "all";
export type Source = "manual" | "eval" | "all";

export type Overview = {
  period: Period;
  source: Source;
  labels: Record<string, number>;
  calls: number;
  in_progress: number;
  funnel: { step: string; count: number }[];
  results: { result: string; words: string; count: number }[];
  agreed_to_pay: number;
  sent_to_team: number;
  promised_jpy: number;
  previous: { calls: number; agreed_to_pay: number; sent_to_team: number; promised_jpy: number } | null;
  per_day: { day: string; agreed: number; sent_to_team: number; other: number }[];
  due_soon: { days: number; count: number; amount_jpy: number };
};

export type CallRow = {
  id: number;
  customer: string;
  customer_ref: string;
  started_at: string;
  sim_started_at: string;
  channel: string;
  label: string;
  agent_version: string;
  source: "manual" | "eval";
  status: "live" | "ended" | "unfinished";
  result: string | null;
  result_words: string;
  summary: string;
};

export type CallDetail = {
  call: CallRow;
  balance_jpy: number | null;
  tone: string | null;
  transcript: { seq: number; who: "AI agent" | "Customer"; text: string; at: string }[];
  moments: { at: string | null; kind: string; text: string }[];
  dates_mentioned: { date: string; spoken: string; turn: number }[];
};

export type CallbackRow = {
  id: number;
  customer: string;
  customer_ref: string;
  due_date: string;
  due_words: string;
  overdue: boolean;
  reason: string;
  reason_words: string;
  customer_words: string;
  status: "requested" | "approved" | "rejected" | "done";
  requested_in_call: number;
  done_in_call: number | null;
  requested_at: string;
};

// --- D-2: customers, payment calendar, needs attention ---

export type CustomerRow = {
  ref: string;
  name: string;
  balance_jpy: number;
  timezone: string;
  badges: string[];
  calls: number;
  last_call: { id: number; at: string; result_words: string; result: string } | null;
};

export type PlanPayment = { due: string; due_words: string; amount_jpy: number; what: string };

export type CustomerDetail = {
  ref: string;
  name: string;
  timezone: string;
  balance_jpy: number;
  paid_total_jpy: number;
  plan_limits: { min_monthly_jpy: number | null; max_months: number | null };
  phones: { mobile: string | null; home: string | null; work: string | null };
  contact: { stop_contact: boolean; no_workplace_contact: boolean };
  notes: string | null;
  payments: { amount_jpy: number; paid_on: string; method: string | null; reference: string | null }[];
  plans: {
    status: string;
    status_words: string;
    kind: string;
    made_on: string;
    total_jpy: number | null;
    payments: PlanPayment[];
    partial_schedule: boolean;
  }[];
  disputes: { open: boolean; reason: string; opened_on: string; resolved_on: string | null }[];
  callbacks: {
    id: number;
    due_words: string;
    status_words: string;
    reason_words: string;
    customer_words: string;
    requested_in_call: number;
  }[];
  calls: CallRow[];
  dates_mentioned: { date: string; spoken: string; call: number; call_at: string }[];
};

export type CalendarDay = {
  day: string;
  day_words: string;
  today: boolean;
  items: {
    type: "payment" | "callback";
    customer: string;
    customer_ref: string;
    what: string;
    amount_jpy: number | null;
    partial_schedule: boolean;
    note: string | null;
  }[];
};

export type Attention = {
  days: number;
  sent_to_team: { customer: string; customer_ref: string; reason_words: string; note: string; call: number; at: string }[];
  open_disputes: { customer: string; customer_ref: string; reason: string; opened_on: string }[];
  stop_contact: { customer: string; customer_ref: string; at: string | null }[];
  unfinished_calls: { customer: string; customer_ref: string; call: number; at: string }[];
};

// --- D-3: quality (owner view) ---

export type Quality = {
  calls: number;
  labels: Record<string, number>;
  versions: string[];
  guard: {
    version: string;
    calls: number;
    checks: number;
    blocked: number;
    then_passed: number;
    then_blocked: number;
    fallbacks: number;
  }[];
  block_reasons: { reason: string; count: number }[];
  rules: { rule: string; allowed: number; blocked: number }[];
  ladder: {
    offers: { rung: string; count: number }[];
    handoffs: { code: string; count: number }[];
    agreed: { rung: string; count: number }[];
  };
  callbacks: { status: string; count: number }[];
};
