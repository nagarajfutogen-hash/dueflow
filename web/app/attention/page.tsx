"use client";

import Link from "next/link";
import { type ReactNode, useEffect, useState } from "react";

import { PageTitle, when } from "@/components/admin";
import { CallbacksPanel } from "@/components/callbacks-panel";
import { ErrorLine } from "@/components/ui";
import { api, type Attention } from "@/lib/api";

function Queue({ title, count, help, children }: { title: string; count: number; help: string; children: ReactNode }) {
  return (
    <section className="flex flex-col rounded-xl border border-slate-200 bg-white shadow-sm">
      <div className="flex flex-col gap-0.5 border-b border-slate-100 px-4 py-3">
        <h2 className="flex items-center gap-2 font-semibold">
          {title}
          <span className="rounded-full bg-slate-900 px-2 py-0.5 text-xs font-medium text-white">{count}</span>
        </h2>
        <span className="text-xs text-slate-500">{help}</span>
      </div>
      {count === 0 ? <p className="px-4 py-6 text-center text-sm text-slate-500">Nothing here.</p> : <ul className="divide-y divide-slate-100">{children}</ul>}
    </section>
  );
}

function Who({ name, refId }: { name: string; refId: string }) {
  return (
    <Link className="font-medium text-blue-700 hover:underline" href={`/customer/?ref=${encodeURIComponent(refId)}`}>
      {name}
    </Link>
  );
}

export default function AttentionPage() {
  const [a, setA] = useState<Attention | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api<Attention>("/admin/attention")
      .then(setA)
      .catch((e: Error) => setError(e.message));
  }, []);

  return (
    <main className="flex flex-col gap-5">
      <PageTitle
        title="Needs attention"
        help="Work for a person: calls sent to the team, disputes, callbacks to approve, and calls that never finished."
      />
      <ErrorLine error={error} />
      <CallbacksPanel />
      {a && (
        <>
          <Queue title="Sent to team" count={a.sent_to_team.length} help={`Last ${a.days} days`}>
            {a.sent_to_team.map((h) => (
              <li key={h.call} className="flex flex-col gap-1 px-4 py-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span>
                    <Who name={h.customer} refId={h.customer_ref} /> <span className="text-slate-600">· {h.reason_words}</span>
                  </span>
                  <Link className="text-sm text-blue-700 hover:underline" href={`/call/?id=${h.call}`}>
                    {when(h.at)}
                  </Link>
                </div>
                <p className="text-sm text-slate-600">Agent&apos;s note: {h.note}</p>
              </li>
            ))}
          </Queue>
          <div className="grid gap-4 lg:grid-cols-3">
            <Queue title="Open disputes" count={a.open_disputes.length} help="No payment talk until resolved">
              {a.open_disputes.map((d, i) => (
                <li key={i} className="flex flex-col px-4 py-3 text-sm">
                  <Who name={d.customer} refId={d.customer_ref} />
                  <span className="text-slate-600">
                    Since {d.opened_on}: {d.reason}
                  </span>
                </li>
              ))}
            </Queue>
            <Queue title="Asked not to be called" count={a.stop_contact.length} help={`Last ${a.days} days`}>
              {a.stop_contact.map((s, i) => (
                <li key={i} className="flex justify-between px-4 py-3 text-sm">
                  <Who name={s.customer} refId={s.customer_ref} />
                  <span className="text-slate-500">{s.at ? when(s.at) : ""}</span>
                </li>
              ))}
            </Queue>
            <Queue title="Calls that never finished" count={a.unfinished_calls.length} help="Browser closed mid-call">
              {a.unfinished_calls.map((u) => (
                <li key={u.call} className="flex justify-between px-4 py-3 text-sm">
                  <Who name={u.customer} refId={u.customer_ref} />
                  <Link className="text-blue-700 hover:underline" href={`/call/?id=${u.call}`}>
                    {when(u.at)}
                  </Link>
                </li>
              ))}
            </Queue>
          </div>
        </>
      )}
    </main>
  );
}
