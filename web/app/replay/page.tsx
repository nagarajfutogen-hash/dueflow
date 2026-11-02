"use client";

import { useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useState } from "react";

import { Card, Chip, ErrorLine, td, th } from "@/components/ui";
import { api, type Estimate, type ReplayRow } from "@/lib/api";

const VERSIONS = ["v1", "v2", "v3"];
const POLL_MS = 3000;

export default function ReplayPage() {
  return (
    <Suspense fallback={<p className="text-sm">Loading…</p>}>
      <ReplayView />
    </Suspense>
  );
}

function ReplayView() {
  const [scenario, setScenario] = useState(useSearchParams().get("scenario") ?? "");
  const [a, setA] = useState("v1");
  const [b, setB] = useState("v3");
  const [estimate, setEstimate] = useState<Estimate | null>(null);
  const [confirmed, setConfirmed] = useState(false);
  const [current, setCurrent] = useState<ReplayRow | null>(null);
  const [recent, setRecent] = useState<ReplayRow[]>([]);
  const [error, setError] = useState<string | null>(null);

  const loadRecent = useCallback(() => {
    api<ReplayRow[]>("/lab/replays")
      .then(setRecent)
      .catch((e: Error) => setError(e.message));
  }, []);
  useEffect(loadRecent, [loadRecent]);

  // Poll the running replay until it is stored as completed or failed.
  useEffect(() => {
    if (current?.status !== "running") return;
    const timer = setInterval(async () => {
      try {
        const r = await api<ReplayRow>(`/lab/replays/${current.id}`);
        setCurrent(r);
        if (r.status !== "running") loadRecent();
      } catch (e) {
        setError((e as Error).message);
      }
    }, POLL_MS);
    return () => clearInterval(timer);
  }, [current, loadRecent]);

  async function getEstimate() {
    setError(null);
    setConfirmed(false);
    try {
      setEstimate(await api<Estimate>(`/lab/replays/estimate?scenario_id=${encodeURIComponent(scenario)}`));
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function start() {
    setError(null);
    try {
      const r = await api<{ id: number }>("/lab/replays", {
        method: "POST",
        body: JSON.stringify({ scenario_id: scenario, version_a: a, version_b: b, confirmed }),
      });
      setCurrent(await api<ReplayRow>(`/lab/replays/${r.id}`));
      setEstimate(null);
      setConfirmed(false);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  const versionSelect = (value: string, set: (v: string) => void, label: string) => (
    <label className="flex items-center gap-1 text-sm">
      {label}
      <select className="rounded border border-slate-300 px-1 py-0.5" value={value} onChange={(e) => set(e.target.value)}>
        {VERSIONS.map((v) => (
          <option key={v}>{v}</option>
        ))}
      </select>
    </label>
  );

  return (
    <div className="flex flex-col gap-4">
      <Card title="Replay a scenario on versions A and B">
        <p className="text-sm text-slate-600">
          One run per version (k=1), then the comparison report. SYNTHETIC. Every replay spends Groq tokens; one runs
          at a time and a few are allowed per day.
        </p>
        <div className="flex flex-wrap items-center gap-3">
          <label className="flex items-center gap-1 text-sm">
            Scenario
            <input
              className="w-72 rounded border border-slate-300 px-2 py-0.5 font-mono"
              value={scenario}
              onChange={(e) => {
                setScenario(e.target.value);
                setEstimate(null);
              }}
              placeholder="base/angry"
            />
          </label>
          {versionSelect(a, setA, "A")}
          {versionSelect(b, setB, "B")}
          <button className="rounded bg-slate-800 px-2 py-1 text-sm text-white disabled:opacity-40" disabled={!scenario} onClick={() => void getEstimate()}>
            Estimate tokens
          </button>
        </div>
        {estimate && (
          <div className="flex flex-col gap-2 rounded bg-slate-100 p-3 text-sm">
            {estimate.total === null ? (
              <p>No stored usage for this scenario: the token cost is unknown.</p>
            ) : (
              <p>
                About <b>{estimate.total.toLocaleString()}</b> Groq tokens ({estimate.per_side?.toLocaleString()} per
                version), from {estimate.n} stored conversation{estimate.n === 1 ? "" : "s"} of this scenario.
              </p>
            )}
            <label className="flex items-center gap-2">
              <input type="checkbox" checked={confirmed} onChange={(e) => setConfirmed(e.target.checked)} />I confirm this
              replay may spend these tokens.
            </label>
            <button
              className="self-start rounded bg-blue-700 px-2 py-1 text-white disabled:opacity-40"
              disabled={!confirmed || current?.status === "running"}
              onClick={() => void start()}
            >
              Start replay
            </button>
          </div>
        )}
        <ErrorLine error={error} />
      </Card>

      {current && <ReplayResult r={current} />}

      <Card title="Recent replays">
        <table className="w-full border-collapse text-sm">
          <thead>
            <tr>
              {["#", "Scenario", "A / B", "Status", "Runs", "Started"].map((h) => (
                <th key={h} className={th}>
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {recent.map((r) => (
              <tr key={r.id} className="cursor-pointer hover:bg-slate-50" onClick={() => setCurrent(r)}>
                <td className={td}>{r.id}</td>
                <td className={`${td} font-mono`}>{r.scenario_id}</td>
                <td className={td}>
                  {r.version_a} / {r.version_b}
                </td>
                <td className={td}>
                  <StatusChip status={r.status} />
                </td>
                <td className={td}>
                  {r.run_a_id ? `#${r.run_a_id}` : "–"} / {r.run_b_id ? `#${r.run_b_id}` : "–"}
                </td>
                <td className={td}>{r.created_at}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  );
}

function StatusChip({ status }: { status: ReplayRow["status"] }) {
  return <Chip tone={status === "completed" ? "green" : status === "failed" ? "red" : "amber"}>{status}</Chip>;
}

function ReplayResult({ r }: { r: ReplayRow }) {
  return (
    <Card title={`Replay #${r.id}: ${r.scenario_id}, ${r.version_a} vs ${r.version_b}`}>
      <div className="flex flex-wrap gap-2 text-sm">
        <StatusChip status={r.status} />
        <Chip>{r.label}</Chip>
        <Chip>k={r.k}</Chip>
      </div>
      {r.status === "running" && <p className="text-sm">Running… (refreshes every few seconds)</p>}
      {r.error && <ErrorLine error={r.error} />}
      {r.report && (
        <pre className="overflow-x-auto whitespace-pre-wrap rounded bg-slate-900 p-3 text-xs text-slate-100">{r.report}</pre>
      )}
    </Card>
  );
}
