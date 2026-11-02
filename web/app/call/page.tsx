"use client";

import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";

import { LabelBadge, ResultPill, when, yen } from "@/components/admin";
import { ErrorLine } from "@/components/ui";
import { api, type CallDetail } from "@/lib/api";

const MOMENT_DOTS: Record<string, string> = {
  identity: "bg-blue-500",
  offer: "bg-slate-500",
  promise: "bg-green-600",
  handoff: "bg-amber-500",
  discount: "bg-red-500",
  payment: "bg-purple-500",
  contact: "bg-red-500",
  tone: "bg-slate-300",
  callback: "bg-purple-500",
  end: "bg-slate-900",
};

export default function CallPage() {
  return (
    <Suspense fallback={<p className="text-sm">Loading…</p>}>
      <CallView />
    </Suspense>
  );
}

function CallView() {
  const id = useSearchParams().get("id");
  const [d, setD] = useState<CallDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const live = d?.call.status === "live";

  useEffect(() => {
    if (!id) return;
    let alive = true;
    const load = () =>
      api<CallDetail>(`/admin/calls/${encodeURIComponent(id)}`)
        .then((v) => alive && setD(v))
        .catch((e: Error) => alive && setError(e.message));
    load();
    // A live call refreshes every 3 seconds until it ends.
    const t = live || d === null ? setInterval(load, 3_000) : undefined;
    return () => {
      alive = false;
      if (t) clearInterval(t);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id, live]);

  if (!id) return <p>No call selected.</p>;
  if (!d) return error ? <ErrorLine error={error} /> : <p className="text-sm text-slate-500">Loading…</p>;
  const c = d.call;

  return (
    <main className="flex flex-col gap-5">
      <div className="flex flex-col gap-2 rounded-lg border border-slate-200 bg-white p-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h1 className="text-2xl font-semibold">{c.customer}</h1>
          <ResultPill result={c.result} status={c.status} words={c.result_words} />
        </div>
        <p className="text-slate-700">{c.summary}</p>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm text-slate-500">
          <span>Customer {c.customer_ref}</span>
          <span>Called {when(c.started_at)}</span>
          <span>{c.channel === "voice" ? "Voice call" : "Chat"}</span>
          {d.balance_jpy !== null && <span>Owed when called: {yen(d.balance_jpy)}</span>}
          {d.tone && <span>Agent's tone: {d.tone}</span>}
          <LabelBadge label={c.label} />
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-[1fr_320px]">
        <section className="flex flex-col gap-3 rounded-lg border border-slate-200 bg-white p-4">
          <h2 className="font-semibold">Conversation {live && <span className="text-sm font-normal text-blue-700">· live</span>}</h2>
          {d.transcript.map((t) => {
            const agent = t.who === "AI agent";
            return (
              <div key={t.seq} className={`flex flex-col ${agent ? "items-start" : "items-end"}`}>
                <span className="text-xs text-slate-500">{t.who}</span>
                <p
                  className={`max-w-[85%] rounded-2xl px-3 py-2 text-sm ${
                    agent ? "rounded-tl-sm bg-slate-100" : "rounded-tr-sm bg-blue-600 text-white"
                  }`}
                >
                  {t.text}
                </p>
              </div>
            );
          })}
          {d.transcript.length === 0 && <p className="text-sm text-slate-500">Nothing said yet.</p>}
        </section>

        <aside className="flex flex-col gap-4">
          <section className="flex flex-col gap-2 rounded-lg border border-slate-200 bg-white p-4">
            <h2 className="font-semibold">Key moments</h2>
            {d.moments.length === 0 && <p className="text-sm text-slate-500">None yet.</p>}
            <ol className="flex flex-col gap-2">
              {d.moments.map((m, i) => (
                <li key={i} className="flex gap-2 text-sm">
                  <span className={`mt-1.5 h-2 w-2 shrink-0 rounded-full ${MOMENT_DOTS[m.kind] ?? "bg-slate-400"}`} />
                  <span>{m.text}</span>
                </li>
              ))}
            </ol>
          </section>
          <section className="flex flex-col gap-2 rounded-lg border border-slate-200 bg-white p-4">
            <h2 className="font-semibold">Dates the customer mentioned</h2>
            <p className="text-xs text-slate-500">Said by the customer; not a payment agreement.</p>
            {d.dates_mentioned.length === 0 ? (
              <p className="text-sm text-slate-500">None.</p>
            ) : (
              <ul className="text-sm">
                {d.dates_mentioned.map((m, i) => (
                  <li key={i}>{m.spoken}</li>
                ))}
              </ul>
            )}
          </section>
        </aside>
      </div>
    </main>
  );
}
