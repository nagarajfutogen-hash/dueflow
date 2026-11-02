import type { ReactNode } from "react";

export function Chip({ children, tone = "slate" }: { children: ReactNode; tone?: "red" | "amber" | "green" | "slate" }) {
  const tones = {
    red: "bg-red-100 text-red-800",
    amber: "bg-amber-100 text-amber-900",
    green: "bg-green-100 text-green-800",
    slate: "bg-slate-200 text-slate-700",
  };
  return <span className={`rounded px-1.5 py-0.5 font-mono text-xs ${tones[tone]}`}>{children}</span>;
}

export function Card({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="flex flex-col gap-2 rounded border border-slate-200 bg-white p-4">
      <h2 className="font-semibold">{title}</h2>
      {children}
    </section>
  );
}

export function ErrorLine({ error }: { error: string | null }) {
  return error ? <p className="text-sm text-red-700">{error}</p> : null;
}

export const th = "border-b border-slate-200 px-2 py-1 text-left font-medium text-slate-600";
export const td = "border-b border-slate-100 px-2 py-1 align-top";
