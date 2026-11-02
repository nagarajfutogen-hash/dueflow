"use client";

import Link from "next/link";
import { type ReactNode, useEffect, useState } from "react";

import { LabelBadge, PageTitle, pct, Segmented, SOURCES } from "@/components/admin";
import { Icon } from "@/components/icons";
import { ErrorLine } from "@/components/ui";
import { api, type Period, type Quality, type Source } from "@/lib/api";

const PERIODS: [Period, string][] = [
  ["today", "Today"],
  ["7d", "7 days"],
  ["30d", "30 days"],
  ["all", "All time"],
];

const LADDER_WORDS: Record<string, string> = {
  pay_in_full: "Pay in full",
  standard_split: "Standard plan (shortest)",
  extended_split: "Extended plan (lowest monthly)",
  customer_plan: "Customer's own plan",
  not_from_the_ladder: "Agreed before v5 (no ladder)",
  below_account_floor: "Below the floor (stated amount too low)",
  counter_limit: "Counter-offer limit reached",
  no_offer_left: "No option left",
};

function Card({ title, help, children }: { title: string; help?: string; children: ReactNode }) {
  return (
    <section className="flex min-w-0 flex-col gap-2 rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div>
        <h2 className="font-semibold">{title}</h2>
        {help && <p className="text-xs text-slate-500">{help}</p>}
      </div>
      {children}
    </section>
  );
}

function Counts({ rows, empty }: { rows: { label: string; count: number }[]; empty: string }) {
  if (rows.length === 0) return <p className="text-sm text-slate-500">{empty}</p>;
  return (
    <ul className="text-sm">
      {rows.map((r) => (
        <li key={r.label} className="flex justify-between border-t border-slate-100 py-1 first:border-0">
          <span>{LADDER_WORDS[r.label] ?? r.label.replaceAll("_", " ")}</span>
          <span className="tabular-nums">{r.count}</span>
        </li>
      ))}
    </ul>
  );
}

const th = "px-2 py-1.5 text-right font-medium";
const td = "px-2 py-1.5 text-right tabular-nums";

export default function QualityPage() {
  const [period, setPeriod] = useState<Period>("all");
  const [source, setSource] = useState<Source>("all");
  const [version, setVersion] = useState("");
  const [q, setQ] = useState<Quality | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api<Quality>(`/admin/quality?period=${period}&source=${source}&version=${version}`)
      .then(setQ)
      .catch((e: Error) => setError(e.message));
  }, [period, source, version]);

  return (
    <main className="flex flex-col gap-5">
      <PageTitle title="Quality" help="How the agent and the safety checks behave. For the owner; technical terms are used here.">
        <div className="flex flex-wrap gap-2">
          <select
            className="rounded-lg border border-slate-300 bg-white px-2 py-1 text-sm"
            value={version}
            onChange={(e) => setVersion(e.target.value)}
          >
            <option value="">All versions</option>
            {["v1", "v2", "v3", "v4", "v5"].map((v) => (
              <option key={v}>{v}</option>
            ))}
          </select>
          <Segmented value={period} options={PERIODS} onChange={setPeriod} />
          <Segmented value={source} options={SOURCES} onChange={setSource} />
        </div>
      </PageTitle>
      <ErrorLine error={error} />

      <div className="grid gap-3 md:grid-cols-2">
        <Link href="/lab/" className="flex items-center gap-3 rounded-xl border border-slate-200 bg-white p-4 shadow-sm hover:border-slate-400">
          <span className="rounded-lg bg-slate-100 p-2">
            <Icon name="lab" />
          </span>
          <span>
            <span className="font-semibold">Failure Lab</span>
            <span className="block text-sm text-slate-600">Failed evaluation scenarios: inspect, promote to regression.</span>
          </span>
        </Link>
        <Link href="/replay/" className="flex items-center gap-3 rounded-xl border border-slate-200 bg-white p-4 shadow-sm hover:border-slate-400">
          <span className="rounded-lg bg-slate-100 p-2">
            <Icon name="calls" />
          </span>
          <span>
            <span className="font-semibold">Replay</span>
            <span className="block text-sm text-slate-600">Run one scenario on two versions and compare (spends tokens).</span>
          </span>
        </Link>
      </div>

      {q && (
        <>
          <p className="-mt-1 flex flex-wrap items-center gap-2 text-sm text-slate-600">
            Based on {q.calls} calls
            {Object.entries(q.labels).map(([label, n]) => (
              <span key={label} className="flex items-center gap-1">
                · {n} <LabelBadge label={label} />
              </span>
            ))}
          </p>

          <Card title="Output guard" help="Every agent reply is checked before it is sent. A block is rewritten once (v4+) or replaced by a fallback line.">
            {q.guard.length === 0 ? (
              <p className="text-sm text-slate-500">No calls.</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="text-slate-600">
                    <tr>
                      <th className="px-2 py-1.5 text-left font-medium">Version</th>
                      <th className={th}>Calls</th>
                      <th className={th}>Replies checked</th>
                      <th className={th}>Blocked</th>
                      <th className={th}>Next reply passed</th>
                      <th className={th}>Blocked again</th>
                      <th className={th}>Fallback lines said</th>
                    </tr>
                  </thead>
                  <tbody>
                    {q.guard.map((g) => (
                      <tr key={g.version} className="border-t border-slate-100">
                        <td className="px-2 py-1.5 font-medium">{g.version}</td>
                        <td className={td}>{g.calls}</td>
                        <td className={td}>{g.checks}</td>
                        <td className={td}>
                          {g.blocked} <span className="text-slate-400">({pct(g.blocked, g.checks)})</span>
                        </td>
                        <td className={td}>{g.then_passed}</td>
                        <td className={td}>{g.then_blocked}</td>
                        <td className={td}>{g.fallbacks}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>

          <div className="grid gap-4 lg:grid-cols-3">
            <Card title="Why replies were blocked">
              <Counts rows={q.block_reasons.map((r) => ({ label: r.reason, count: r.count }))} empty="No blocks." />
            </Card>
            <Card title="Payment ladder (v5)" help="Offers the engine made, and why it handed off.">
              <h3 className="text-xs font-semibold uppercase text-slate-500">Offers</h3>
              <Counts rows={q.ladder.offers.map((r) => ({ label: r.rung, count: r.count }))} empty="No v5 offers yet." />
              <h3 className="mt-2 text-xs font-semibold uppercase text-slate-500">Handoffs</h3>
              <Counts rows={q.ladder.handoffs.map((r) => ({ label: r.code, count: r.count }))} empty="None." />
              <h3 className="mt-2 text-xs font-semibold uppercase text-slate-500">Agreed</h3>
              <Counts rows={q.ladder.agreed.map((r) => ({ label: r.rung, count: r.count }))} empty="No agreements." />
            </Card>
            <Card title="Callbacks (v5)">
              <Counts rows={q.callbacks.map((r) => ({ label: r.status, count: r.count }))} empty="No callbacks requested." />
            </Card>
          </div>

          <Card title="Every policy rule" help="Allowed and blocked decisions, from the audit log. Most-blocked first.">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="text-slate-600">
                  <tr>
                    <th className="px-2 py-1.5 text-left font-medium">Rule</th>
                    <th className={th}>Allowed</th>
                    <th className={th}>Blocked</th>
                    <th className={th}>Blocked share</th>
                  </tr>
                </thead>
                <tbody>
                  {q.rules.map((r) => (
                    <tr key={r.rule} className="border-t border-slate-100">
                      <td className="px-2 py-1.5 font-mono text-xs">{r.rule}</td>
                      <td className={td}>{r.allowed}</td>
                      <td className={td}>{r.blocked}</td>
                      <td className={td}>{pct(r.blocked, r.allowed + r.blocked)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-xs text-slate-500">
              A block is not a failure by itself: it is the engine doing its job (for example identity_verification blocks
              include “no date of birth said yet”).
            </p>
          </Card>
        </>
      )}
    </main>
  );
}
