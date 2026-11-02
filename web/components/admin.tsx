// Shared pieces for the admin dashboard: plain words, big numbers, honest labels.
import type { ReactNode } from "react";

import { Icon, type IconName } from "@/components/icons";

export const yen = (n: number) => `¥${n.toLocaleString("en-US")}`;

export const when = (iso: string) =>
  new Date(iso).toLocaleString("en-GB", { dateStyle: "medium", timeStyle: "short" });

export const pct = (part: number, whole: number) => (whole ? `${Math.round((part / whole) * 100)}%` : "–");

const RESULT_TONES: Record<string, string> = {
  promise: "bg-green-100 text-green-800",
  handoff: "bg-amber-100 text-amber-900",
  disconnected: "bg-red-100 text-red-800",
  live: "bg-blue-100 text-blue-800",
  callback_scheduled: "bg-purple-100 text-purple-800",
  unfinished: "bg-slate-200 text-slate-700",
};

export function ResultPill({ result, status, words }: { result: string | null; status: string; words: string }) {
  const tone = RESULT_TONES[result ?? status] ?? "bg-slate-100 text-slate-700";
  return (
    <span className={`inline-flex items-center gap-1 whitespace-nowrap rounded-full px-2.5 py-0.5 text-xs font-medium ${tone}`}>
      {status === "live" && <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-blue-600" />}
      {words}
    </span>
  );
}

export function LabelBadge({ label }: { label: string }) {
  return (
    <span
      title="SYNTHETIC = simulated customers. HUMAN TEST = people role-playing a customer. No real customers yet."
      className="rounded border border-amber-300 bg-amber-50 px-1.5 py-0.5 text-[10px] font-semibold tracking-wide text-amber-900"
    >
      {label.replace("_", " ")}
    </span>
  );
}

export function StatCard({
  title,
  value,
  note,
  icon,
  tone = "slate",
  change,
}: {
  title: string;
  value: ReactNode;
  note?: ReactNode;
  icon?: IconName;
  tone?: "slate" | "green" | "amber" | "blue" | "purple";
  change?: ReactNode;
}) {
  const tones = {
    slate: "bg-slate-100 text-slate-700",
    green: "bg-green-100 text-green-700",
    amber: "bg-amber-100 text-amber-700",
    blue: "bg-blue-100 text-blue-700",
    purple: "bg-purple-100 text-purple-700",
  };
  return (
    <div className="flex h-full flex-col gap-3 rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-center justify-between gap-2">
        <span className="text-sm font-medium text-slate-600">{title}</span>
        {icon && (
          <span className={`rounded-lg p-2 ${tones[tone]}`}>
            <Icon name={icon} className="h-4 w-4" />
          </span>
        )}
      </div>
      <span className="text-3xl font-semibold tabular-nums">{value}</span>
      <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
        {change}
        {note && <span>{note}</span>}
      </div>
    </div>
  );
}

// "▲2 vs previous 7 days": the change against the window just before this one.
// upIsGood=false for counts where more is worse (calls sent to the team): colours flip.
export function Change({
  now,
  before,
  label,
  money = false,
  upIsGood = true,
}: {
  now: number;
  before: number;
  label: string;
  money?: boolean;
  upIsGood?: boolean;
}) {
  const diff = now - before;
  const text = money ? yen(Math.abs(diff)) : Math.abs(diff).toLocaleString("en-US");
  const good = upIsGood ? diff > 0 : diff < 0;
  const tone = diff === 0 ? "text-slate-500" : good ? "text-green-700" : "text-red-700";
  return (
    <span className={`font-medium ${tone}`} title={`Previous period: ${money ? yen(before) : before}`}>
      {diff > 0 ? "▲" : diff < 0 ? "▼" : "="} {diff === 0 ? "same" : text} vs {label}
    </span>
  );
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
}: {
  value: T;
  options: [T, string][];
  onChange: (v: T) => void;
}) {
  return (
    <div className="inline-flex rounded-lg border border-slate-300 bg-white p-0.5 text-sm">
      {options.map(([v, text]) => (
        <button
          key={v}
          onClick={() => onChange(v)}
          className={`rounded-md px-3 py-1 ${v === value ? "bg-slate-900 text-white" : "text-slate-700 hover:bg-slate-100"}`}
        >
          {text}
        </button>
      ))}
    </div>
  );
}

export const SOURCES: ["manual" | "eval" | "all", string][] = [
  ["manual", "Test calls"],
  ["eval", "Evaluation runs"],
  ["all", "Everything"],
];

export function PageTitle({ title, help, children }: { title: string; help: string; children?: ReactNode }) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-3">
      <div>
        <h1 className="text-2xl font-semibold">{title}</h1>
        <p className="text-sm text-slate-600">{help}</p>
      </div>
      {children}
    </div>
  );
}

// Customer badges (Customers and customer pages).
const BADGE_TONES: Record<string, string> = {
  "Open dispute": "bg-amber-100 text-amber-900",
  "Don't call": "bg-red-100 text-red-800",
  "Not at work": "bg-slate-200 text-slate-700",
  "Active plan": "bg-green-100 text-green-800",
  "Callback waiting": "bg-blue-100 text-blue-800",
  "Callback approved": "bg-purple-100 text-purple-800",
};

export function Badge({ text }: { text: string }) {
  return (
    <span className={`whitespace-nowrap rounded-full px-2 py-0.5 text-xs font-medium ${BADGE_TONES[text] ?? "bg-slate-100"}`}>
      {text}
    </span>
  );
}
