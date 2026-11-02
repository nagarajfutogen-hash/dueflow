"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { LabelBadge, PageTitle, ResultPill, Segmented, SOURCES, when } from "@/components/admin";
import { ErrorLine } from "@/components/ui";
import { api, type CallRow, type Source } from "@/lib/api";

export default function CallsPage() {
  const [source, setSource] = useState<Source>("manual");
  const [query, setQuery] = useState("");
  const [rows, setRows] = useState<CallRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setRows(null);
    api<CallRow[]>(`/admin/calls?source=${source}`)
      .then(setRows)
      .catch((e: Error) => setError(e.message));
  }, [source]);

  const q = query.trim().toLowerCase();
  const shown = rows?.filter(
    (c) => !q || `${c.customer} ${c.customer_ref} ${c.result_words} ${c.summary}`.toLowerCase().includes(q),
  );

  return (
    <main className="flex flex-col gap-5">
      <PageTitle title="All calls" help="Every call, newest first (latest 200). Open one to read the whole conversation.">
        <div className="flex flex-wrap gap-2">
          <input
            placeholder="Search customer or result"
            className="rounded-lg border border-slate-300 px-3 py-1 text-sm"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <Segmented value={source} options={SOURCES} onChange={setSource} />
        </div>
      </PageTitle>
      <ErrorLine error={error} />
      {!rows && !error && <p className="text-sm text-slate-500">Loading…</p>}
      {shown && (
        <div className="overflow-x-auto rounded-lg border border-slate-200 bg-white">
          <table className="w-full text-sm">
            <thead className="bg-slate-50 text-left text-slate-600">
              <tr>
                <th className="px-3 py-2 font-medium">When</th>
                <th className="px-3 py-2 font-medium">Customer</th>
                <th className="px-3 py-2 font-medium">Result</th>
                <th className="px-3 py-2 font-medium">What happened</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((c) => (
                <tr key={c.id} className="border-t border-slate-100 align-top hover:bg-slate-50">
                  <td className="whitespace-nowrap px-3 py-2 text-slate-600">{when(c.started_at)}</td>
                  <td className="px-3 py-2">
                    <Link className="font-medium text-blue-700 hover:underline" href={`/call/?id=${c.id}`}>
                      {c.customer}
                    </Link>
                    <div className="flex items-center gap-1 text-xs text-slate-500">
                      {c.customer_ref} <LabelBadge label={c.label} />
                    </div>
                  </td>
                  <td className="px-3 py-2">
                    <ResultPill result={c.result} status={c.status} words={c.result_words} />
                  </td>
                  <td className="px-3 py-2 text-slate-700">{c.summary}</td>
                </tr>
              ))}
              {shown.length === 0 && (
                <tr>
                  <td colSpan={4} className="px-3 py-6 text-center text-slate-500">
                    No calls found.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}
    </main>
  );
}
