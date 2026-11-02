"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { Badge, PageTitle, when, yen } from "@/components/admin";
import { ErrorLine } from "@/components/ui";
import { api, type CustomerRow } from "@/lib/api";

export default function CustomersPage() {
  const [rows, setRows] = useState<CustomerRow[] | null>(null);
  const [query, setQuery] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api<CustomerRow[]>("/admin/customers")
      .then(setRows)
      .catch((e: Error) => setError(e.message));
  }, []);

  const q = query.trim().toLowerCase();
  const shown = rows?.filter((c) => !q || `${c.name} ${c.ref} ${c.badges.join(" ")}`.toLowerCase().includes(q));

  return (
    <main className="flex flex-col gap-5">
      <PageTitle title="Customers" help="Every test customer: what they owe, how the last call went, and anything to watch.">
        <input
          placeholder="Search name, ID or badge"
          className="rounded-lg border border-slate-300 px-3 py-1 text-sm"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </PageTitle>
      <ErrorLine error={error} />
      {!rows && !error && <p className="text-sm text-slate-500">Loading…</p>}
      {shown && (
        <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white shadow-sm">
          <table className="w-full text-sm">
            <thead className="bg-slate-50 text-left text-slate-600">
              <tr>
                <th className="px-3 py-2 font-medium">Customer</th>
                <th className="px-3 py-2 text-right font-medium">Owes</th>
                <th className="px-3 py-2 font-medium">Last call</th>
                <th className="px-3 py-2 font-medium">To watch</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((c) => (
                <tr key={c.ref} className="border-t border-slate-100 align-top hover:bg-slate-50">
                  <td className="px-3 py-2">
                    <Link className="font-medium text-blue-700 hover:underline" href={`/customer/?ref=${encodeURIComponent(c.ref)}`}>
                      {c.name}
                    </Link>
                    <div className="text-xs text-slate-500">
                      {c.ref} · {c.calls} call{c.calls === 1 ? "" : "s"}
                    </div>
                  </td>
                  <td className="whitespace-nowrap px-3 py-2 text-right tabular-nums">{yen(c.balance_jpy)}</td>
                  <td className="px-3 py-2 text-slate-700">
                    {c.last_call ? (
                      <>
                        {c.last_call.result_words}
                        <div className="text-xs text-slate-500">{when(c.last_call.at)}</div>
                      </>
                    ) : (
                      <span className="text-slate-400">Not called yet</span>
                    )}
                  </td>
                  <td className="px-3 py-2">
                    <div className="flex flex-wrap gap-1">
                      {c.badges.map((b) => (
                        <Badge key={b} text={b} />
                      ))}
                    </div>
                  </td>
                </tr>
              ))}
              {shown.length === 0 && (
                <tr>
                  <td colSpan={4} className="px-3 py-6 text-center text-slate-500">
                    No customers found.
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
