"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { PageTitle, yen } from "@/components/admin";
import { ErrorLine } from "@/components/ui";
import { api, type CalendarDay } from "@/lib/api";

function monthStart(offset: number): Date {
  const now = new Date();
  return new Date(now.getFullYear(), now.getMonth() + offset, 1);
}

const iso = (d: Date) =>
  `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;

export default function CalendarPage() {
  const [offset, setOffset] = useState(0);
  const [days, setDays] = useState<CalendarDay[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const start = monthStart(offset);
  const length = new Date(start.getFullYear(), start.getMonth() + 1, 0).getDate();
  const label = start.toLocaleDateString("en-GB", { month: "long", year: "numeric" });

  useEffect(() => {
    setDays(null);
    api<CalendarDay[]>(`/admin/calendar?start=${iso(start)}&days=${length}`)
      .then(setDays)
      .catch((e: Error) => setError(e.message));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [offset]);

  const payments = days?.flatMap((d) => d.items.filter((i) => i.type === "payment")) ?? [];
  const due = payments.reduce((sum, i) => sum + (i.amount_jpy ?? 0), 0);

  return (
    <main className="flex flex-col gap-5">
      <PageTitle title="Calendar" help="Payments due on active plans, and callbacks, day by day.">
        <div className="flex items-center gap-2">
          <button className="rounded-lg border border-slate-300 bg-white px-3 py-1 text-sm" onClick={() => setOffset(offset - 1)}>
            ←
          </button>
          <span className="w-36 text-center font-medium">{label}</span>
          <button className="rounded-lg border border-slate-300 bg-white px-3 py-1 text-sm" onClick={() => setOffset(offset + 1)}>
            →
          </button>
        </div>
      </PageTitle>
      <ErrorLine error={error} />
      {days && (
        <p className="-mt-2 text-sm text-slate-600">
          {payments.length} payment{payments.length === 1 ? "" : "s"} due this month ({yen(due)}) ·{" "}
          {days.reduce((n, d) => n + d.items.filter((i) => i.type === "callback").length, 0)} callbacks. Nothing marks a plan
          kept or broken automatically: a passed date only shows whether a payment is on file since.
        </p>
      )}
      {days && days.length === 0 && (
        <p className="rounded-lg border border-dashed border-slate-300 bg-white p-8 text-center text-slate-500">
          Nothing due in {label}.
        </p>
      )}
      <div className="flex flex-col gap-3">
        {days?.map((d) => (
          <section
            key={d.day}
            className={`flex flex-col rounded-xl border bg-white shadow-sm ${d.today ? "border-blue-400" : "border-slate-200"}`}
          >
            <h2 className="border-b border-slate-100 px-4 py-2 font-semibold">
              {d.day_words}
              {d.today && <span className="ml-2 text-sm font-normal text-blue-700">today</span>}
            </h2>
            <ul className="divide-y divide-slate-100">
              {d.items.map((i, k) => (
                <li key={k} className="flex flex-wrap items-center justify-between gap-2 px-4 py-2 text-sm">
                  <span className="flex items-center gap-2">
                    <span
                      className={`rounded-full px-2 py-0.5 text-xs font-medium ${
                        i.type === "payment" ? "bg-green-100 text-green-800" : "bg-purple-100 text-purple-800"
                      }`}
                    >
                      {i.type === "payment" ? "Payment" : "Callback"}
                    </span>
                    <Link className="font-medium text-blue-700 hover:underline" href={`/customer/?ref=${encodeURIComponent(i.customer_ref)}`}>
                      {i.customer}
                    </Link>
                    <span className="text-slate-600">{i.what}</span>
                  </span>
                  <span className="flex items-center gap-3">
                    {i.note && (
                      <span className={i.note.startsWith("Due date passed") ? "text-red-700" : "text-slate-500"}>{i.note}</span>
                    )}
                    {i.amount_jpy !== null && <span className="tabular-nums font-medium">{yen(i.amount_jpy)}</span>}
                  </span>
                </li>
              ))}
            </ul>
          </section>
        ))}
      </div>
    </main>
  );
}
