"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { Icon } from "@/components/icons";
import { api, type CallbackRow } from "@/lib/api";

// Callbacks customers asked for. Nobody is called until the owner approves one here; after
// that the auto dialer calls on the day (the normal contact checks still decide).
export function CallbacksPanel() {
  const [rows, setRows] = useState<CallbackRow[] | null>(null);
  const [busy, setBusy] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    api<CallbackRow[]>("/admin/callbacks?status=requested")
      .then(setRows)
      .catch((e: Error) => setError(e.message));
  }, []);

  useEffect(() => {
    load();
    const t = setInterval(load, 10_000);
    return () => clearInterval(t);
  }, [load]);

  async function decide(id: number, action: "approve" | "reject") {
    setBusy(id);
    setError(null);
    try {
      await api(`/admin/callbacks/${id}/${action}`, { method: "POST" });
      load();
    } catch (e) {
      setError((e as Error).message.replace(/^\d+: /, ""));
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="flex flex-col rounded-xl border border-slate-200 bg-white shadow-sm">
      <div className="flex items-center justify-between border-b border-slate-100 px-4 py-3">
        <h2 className="flex items-center gap-2 font-semibold">
          <Icon name="calendar" className="h-4 w-4 text-slate-500" />
          Callbacks to approve
          {rows && rows.length > 0 && (
            <span className="rounded-full bg-blue-600 px-2 py-0.5 text-xs font-medium text-white">{rows.length}</span>
          )}
        </h2>
        <span className="text-xs text-slate-500">Nobody is called until you approve</span>
      </div>
      {error && <p className="px-4 pt-3 text-sm text-red-700">{error}</p>}
      {rows && rows.length === 0 && <p className="px-4 py-8 text-center text-sm text-slate-500">No callback requests waiting.</p>}
      <ul className="divide-y divide-slate-100">
        {rows?.map((c) => (
          <li key={c.id} className="flex flex-wrap items-center justify-between gap-3 px-4 py-3">
            <div className="flex min-w-0 flex-col gap-0.5">
              <span className="font-medium">{c.customer}</span>
              <span className="text-sm text-slate-700">
                Call back on <b>{c.due_words}</b>
                {c.overdue && <span className="ml-1 text-red-700">(day has passed)</span>} · {c.reason_words}
              </span>
              <span className="text-xs text-slate-500">
                They said “{c.customer_words}” ·{" "}
                <Link className="text-blue-700 hover:underline" href={`/call/?id=${c.requested_in_call}`}>
                  see the call
                </Link>
              </span>
            </div>
            <div className="flex gap-2">
              <button
                disabled={busy === c.id || c.overdue}
                onClick={() => decide(c.id, "approve")}
                title={c.overdue ? "The day has passed: reject it instead" : "The customer will be called on this day"}
                className="rounded-lg bg-slate-900 px-3 py-1.5 text-sm text-white disabled:opacity-40"
              >
                Approve
              </button>
              <button
                disabled={busy === c.id}
                onClick={() => decide(c.id, "reject")}
                className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm hover:bg-slate-50 disabled:opacity-40"
              >
                Reject
              </button>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
