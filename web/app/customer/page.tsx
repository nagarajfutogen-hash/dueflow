"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { type ReactNode, Suspense, useEffect, useState } from "react";

import { Badge, ResultPill, when, yen } from "@/components/admin";
import { ErrorLine } from "@/components/ui";
import { api, type CustomerDetail } from "@/lib/api";

export default function CustomerPage() {
  return (
    <Suspense fallback={<p className="text-sm">Loading…</p>}>
      <CustomerView />
    </Suspense>
  );
}

function Section({ title, children, empty }: { title: string; children: ReactNode; empty?: boolean }) {
  return (
    <section className="flex flex-col gap-2 rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <h2 className="font-semibold">{title}</h2>
      {empty ? <p className="text-sm text-slate-500">None.</p> : children}
    </section>
  );
}

function CustomerView() {
  const ref = useSearchParams().get("ref");
  const [c, setC] = useState<CustomerDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!ref) return;
    api<CustomerDetail>(`/admin/customers/${encodeURIComponent(ref)}`)
      .then(setC)
      .catch((e: Error) => setError(e.message));
  }, [ref]);

  if (!ref) return <p>No customer selected.</p>;
  if (!c) return error ? <ErrorLine error={error} /> : <p className="text-sm text-slate-500">Loading…</p>;

  const badges = [
    ...(c.disputes.some((d) => d.open) ? ["Open dispute"] : []),
    ...(c.contact.stop_contact ? ["Don't call"] : []),
    ...(c.contact.no_workplace_contact ? ["Not at work"] : []),
    ...(c.plans.some((p) => p.status === "active") ? ["Active plan"] : []),
  ];
  const phones = Object.entries(c.phones).filter(([, v]) => v);

  return (
    <main className="flex flex-col gap-5">
      <div className="flex flex-col gap-3 rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h1 className="text-2xl font-semibold">{c.name}</h1>
            <p className="text-sm text-slate-500">
              {c.ref} · {c.timezone} · {phones.map(([k, v]) => `${k} ${v}`).join(" · ") || "no phone"}
            </p>
          </div>
          <div className="flex flex-wrap gap-1">
            {badges.map((b) => (
              <Badge key={b} text={b} />
            ))}
          </div>
        </div>
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <Stat label="Owes now" value={yen(c.balance_jpy)} />
          <Stat label="Paid so far" value={yen(c.paid_total_jpy)} />
          <Stat
            label="Lowest monthly offer"
            value={c.plan_limits.min_monthly_jpy ? yen(c.plan_limits.min_monthly_jpy) : "No minimum"}
          />
          <Stat label="Most months offered" value={c.plan_limits.max_months ? String(c.plan_limits.max_months) : "Up to 6"} />
        </div>
        {c.notes && <p className="rounded-lg bg-slate-50 px-3 py-2 text-sm text-slate-700">Notes: {c.notes}</p>}
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Section title="Payment plans" empty={c.plans.length === 0}>
          {c.plans.map((p, i) => (
            <div key={i} className="flex flex-col gap-1 border-t border-slate-100 pt-2 first:border-0 first:pt-0">
              <div className="flex justify-between text-sm">
                <span className="font-medium">
                  {p.status_words} {p.kind === "pay_in_full" ? "full payment" : "plan"}
                </span>
                <span className="text-slate-500">agreed {when(p.made_on)}</span>
              </div>
              <ul className="text-sm text-slate-700">
                {p.payments.map((x, j) => (
                  <li key={j} className="flex justify-between">
                    <span>
                      {x.what} · {x.due_words}
                    </span>
                    <span className="tabular-nums">{yen(x.amount_jpy)}</span>
                  </li>
                ))}
              </ul>
              {p.partial_schedule && (
                <p className="text-xs text-slate-500">Older plan: only monthly payments were stored (no down payment date).</p>
              )}
            </div>
          ))}
        </Section>

        <Section title="Payments received" empty={c.payments.length === 0}>
          <ul className="text-sm">
            {c.payments.map((p, i) => (
              <li key={i} className="flex justify-between border-t border-slate-100 py-1 first:border-0">
                <span>
                  {p.paid_on} · {p.method?.replace("_", " ") ?? "method not recorded"}
                  {p.reference && <span className="text-slate-500"> · {p.reference}</span>}
                </span>
                <span className="tabular-nums">{yen(p.amount_jpy)}</span>
              </li>
            ))}
          </ul>
        </Section>

        <Section title="Callbacks" empty={c.callbacks.length === 0}>
          <ul className="text-sm">
            {c.callbacks.map((cb) => (
              <li key={cb.id} className="border-t border-slate-100 py-1 first:border-0">
                <b>{cb.due_words}</b> · {cb.status_words} · {cb.reason_words}
                <span className="text-slate-500"> · they said “{cb.customer_words}”</span>
              </li>
            ))}
          </ul>
        </Section>

        <Section title="Disputes" empty={c.disputes.length === 0}>
          <ul className="text-sm">
            {c.disputes.map((d, i) => (
              <li key={i} className="border-t border-slate-100 py-1 first:border-0">
                <b>{d.open ? "Open" : "Resolved"}</b> since {d.opened_on}: {d.reason}
              </li>
            ))}
          </ul>
        </Section>

        <Section title="Dates the customer mentioned" empty={c.dates_mentioned.length === 0}>
          <p className="text-xs text-slate-500">Said by the customer; not a payment agreement.</p>
          <ul className="text-sm">
            {c.dates_mentioned.map((d, i) => (
              <li key={i}>
                {d.spoken} ·{" "}
                <Link className="text-blue-700 hover:underline" href={`/call/?id=${d.call}`}>
                  call of {when(d.call_at)}
                </Link>
              </li>
            ))}
          </ul>
        </Section>
      </div>

      <Section title={`Calls (${c.calls.length})`} empty={c.calls.length === 0}>
        <ol className="flex flex-col">
          {c.calls.map((call) => (
            <li key={call.id} className="flex flex-col gap-1 border-t border-slate-100 py-2 first:border-0">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <Link className="font-medium text-blue-700 hover:underline" href={`/call/?id=${call.id}`}>
                  {when(call.started_at)}
                </Link>
                <ResultPill result={call.result} status={call.status} words={call.result_words} />
              </div>
              <p className="text-sm text-slate-700">{call.summary}</p>
            </li>
          ))}
        </ol>
      </Section>
    </main>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-slate-50 px-3 py-2">
      <div className="text-xs text-slate-500">{label}</div>
      <div className="text-lg font-semibold tabular-nums">{value}</div>
    </div>
  );
}
