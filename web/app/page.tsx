"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { Change, LabelBadge, PageTitle, pct, ResultPill, Segmented, SOURCES, StatCard, when, yen } from "@/components/admin";
import { CallbacksPanel } from "@/components/callbacks-panel";
import { Icon } from "@/components/icons";
import { ErrorLine } from "@/components/ui";
import { api, type CallRow, type Overview, type Period, type Source } from "@/lib/api";

const PERIODS: [Period, string][] = [
  ["today", "Today"],
  ["7d", "Last 7 days"],
  ["30d", "Last 30 days"],
  ["all", "All time"],
];
const PREVIOUS_LABEL: Record<Period, string> = {
  today: "yesterday",
  "7d": "previous 7 days",
  "30d": "previous 30 days",
  all: "",
};

const shortDay = (iso: string) =>
  new Date(`${iso}T00:00:00`).toLocaleDateString("en-GB", { day: "numeric", month: "short" });

export default function TodayPage() {
  const [period, setPeriod] = useState<Period>("7d");
  const [source, setSource] = useState<Source>("manual");
  const [o, setO] = useState<Overview | null>(null);
  const [calls, setCalls] = useState<CallRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    const load = () => {
      api<Overview>(`/admin/overview?period=${period}&source=${source}`)
        .then((v) => alive && setO(v))
        .catch((e: Error) => alive && setError(e.message));
      api<CallRow[]>(`/admin/calls?source=${source}`)
        .then((v) => alive && setCalls(v))
        .catch((e: Error) => alive && setError(e.message));
    };
    load();
    const t = setInterval(load, 10_000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [period, source]);

  const prev = o?.previous;
  const prevLabel = PREVIOUS_LABEL[period];
  const attention = calls?.filter((c) => c.result === "handoff").slice(0, 6) ?? [];
  const recent = calls?.slice(0, 6) ?? [];

  return (
    <main className="flex flex-col gap-6">
      <PageTitle title="Today" help="How the AI agent's calls are going. Updates every 10 seconds.">
        <div className="flex flex-wrap gap-2">
          <Segmented value={period} options={PERIODS} onChange={setPeriod} />
          <Segmented value={source} options={SOURCES} onChange={setSource} />
        </div>
      </PageTitle>
      <ErrorLine error={error} />

      {o && (
        <>
          <p className="-mt-3 flex flex-wrap items-center gap-2 text-sm text-slate-600">
            Based on {o.calls} call{o.calls === 1 ? "" : "s"}
            {Object.entries(o.labels).map(([label, n]) => (
              <span key={label} className="flex items-center gap-1">
                · {n} <LabelBadge label={label} />
              </span>
            ))}
            {o.calls > 0 && o.calls < 30 && <span>· small sample: read the rates as a rough direction</span>}
          </p>

          <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
            <StatCard
              title="Calls made"
              icon="phone"
              value={o.calls}
              change={prev && <Change now={o.calls} before={prev.calls} label={prevLabel} />}
              note={
                o.in_progress ? (
                  <Link href="/live/" className="font-medium text-blue-700">
                    {o.in_progress} live now
                  </Link>
                ) : undefined
              }
            />
            <StatCard
              title="Agreed to pay"
              icon="check"
              tone="green"
              value={o.agreed_to_pay}
              note={`${pct(o.agreed_to_pay, o.calls)} of calls`}
              change={prev && <Change now={o.agreed_to_pay} before={prev.agreed_to_pay} label={prevLabel} />}
            />
            <StatCard
              title="Sent to team"
              icon="team"
              tone="amber"
              value={o.sent_to_team}
              note={`${pct(o.sent_to_team, o.calls)} of calls`}
              change={prev && <Change now={o.sent_to_team} before={prev.sent_to_team} label={prevLabel} upIsGood={false} />}
            />
            <StatCard
              title="Amount promised"
              icon="yen"
              tone="purple"
              value={yen(o.promised_jpy)}
              change={prev && <Change now={o.promised_jpy} before={prev.promised_jpy} label={prevLabel} money />}
            />
            <StatCard
              title={`Due in next ${o.due_soon.days} days`}
              icon="calendar"
              tone="blue"
              value={o.due_soon.count}
              note={`${yen(o.due_soon.amount_jpy)} on active agreements`}
            />
          </div>

          <section className="grid gap-4 lg:grid-cols-3">
            <div className="flex min-w-0 flex-col gap-2 rounded-xl border border-slate-200 bg-white p-4 shadow-sm lg:col-span-2">
              <h2 className="font-semibold">Calls per day</h2>
              {o.per_day.length === 0 ? (
                <p className="py-16 text-center text-sm text-slate-500">No calls in this period.</p>
              ) : (
                <div className="h-[260px] w-full min-w-0">
                <ResponsiveContainer width="100%" height="100%" key={`${period}-${source}`}>
                  <BarChart data={o.per_day} margin={{ left: -20, right: 8 }} maxBarSize={48}>
                    <CartesianGrid vertical={false} stroke="#e2e8f0" />
                    <XAxis dataKey="day" tickFormatter={shortDay} tick={{ fontSize: 12 }} />
                    <YAxis allowDecimals={false} tick={{ fontSize: 12 }} />
                    <Tooltip labelFormatter={(d: string) => shortDay(d)} />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <Bar dataKey="agreed" name="Agreed to pay" stackId="a" fill="#16a34a" isAnimationActive={false} />
                    <Bar dataKey="sent_to_team" name="Sent to team" stackId="a" fill="#f59e0b" isAnimationActive={false} />
                    <Bar dataKey="other" name="Other results" stackId="a" fill="#94a3b8" radius={[4, 4, 0, 0]} isAnimationActive={false} />
                  </BarChart>
                </ResponsiveContainer>
                </div>
              )}
            </div>

            <div className="flex flex-col gap-3 rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
              <h2 className="font-semibold">From call to agreement</h2>
              {o.funnel.map((s) => (
                <div key={s.step} className="flex flex-col gap-1">
                  <div className="flex justify-between text-sm">
                    <span>{s.step}</span>
                    <span className="tabular-nums text-slate-600">
                      {s.count} <span className="text-slate-400">({pct(s.count, o.calls)})</span>
                    </span>
                  </div>
                  <div className="h-2 rounded-full bg-slate-100">
                    <div
                      className="h-2 rounded-full bg-slate-800"
                      style={{ width: `${o.calls ? (s.count / o.calls) * 100 : 0}%` }}
                    />
                  </div>
                </div>
              ))}
              <h3 className="mt-2 text-sm font-semibold">How calls ended</h3>
              <ul className="flex flex-col gap-1 text-sm">
                {o.results.map((r) => (
                  <li key={r.result} className="flex justify-between">
                    <span className="text-slate-700">{r.words}</span>
                    <span className="tabular-nums text-slate-600">{r.count}</span>
                  </li>
                ))}
                {o.results.length === 0 && <li className="text-slate-500">No calls yet.</li>}
              </ul>
            </div>
          </section>
        </>
      )}

      <CallbacksPanel />

      <section className="grid gap-4 lg:grid-cols-2">
        <Panel title="Needs attention" icon="alert" link="/calls/" empty="Nothing waiting for the team.">
          {attention.map((c) => (
            <CallLine key={c.id} c={c} />
          ))}
        </Panel>
        <Panel title="Recent calls" icon="phone" link="/calls/" empty="No calls yet.">
          {recent.map((c) => (
            <CallLine key={c.id} c={c} />
          ))}
        </Panel>
      </section>
    </main>
  );
}

function Panel({
  title,
  icon,
  link,
  empty,
  children,
}: {
  title: string;
  icon: "alert" | "phone";
  link: string;
  empty: string;
  children: React.ReactNode[];
}) {
  return (
    <div className="flex flex-col rounded-xl border border-slate-200 bg-white shadow-sm">
      <div className="flex items-center justify-between border-b border-slate-100 px-4 py-3">
        <h2 className="flex items-center gap-2 font-semibold">
          <Icon name={icon} className="h-4 w-4 text-slate-500" />
          {title}
        </h2>
        <Link href={link} className="text-sm text-blue-700 hover:underline">
          See all
        </Link>
      </div>
      {children.length === 0 ? (
        <p className="px-4 py-8 text-center text-sm text-slate-500">{empty}</p>
      ) : (
        <ul className="divide-y divide-slate-100">{children}</ul>
      )}
    </div>
  );
}

function CallLine({ c }: { c: CallRow }) {
  return (
    <li>
      <Link href={`/call/?id=${c.id}`} className="flex flex-col gap-1 px-4 py-3 hover:bg-slate-50">
        <div className="flex items-center justify-between gap-2">
          <span className="font-medium">{c.customer}</span>
          <ResultPill result={c.result} status={c.status} words={c.result_words} />
        </div>
        <p className="line-clamp-2 text-sm text-slate-600">{c.summary}</p>
        <span className="text-xs text-slate-400">{when(c.started_at)}</span>
      </Link>
    </li>
  );
}
