"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { Card, Chip, ErrorLine, td, th } from "@/components/ui";
import { api, type Failure } from "@/lib/api";

const VERSIONS = ["", "v1", "v2", "v3", "v4", "v5"];
const LABELS = ["", "SYNTHETIC", "HUMAN_TEST"];

export default function FailuresPage() {
  const [rows, setRows] = useState<Failure[] | null>(null);
  const [categories, setCategories] = useState<string[]>([]);
  const [filter, setFilter] = useState({ category: "", version: "", label: "" });
  const [history, setHistory] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const load = useCallback(async () => {
    setError(null);
    try {
      const q = new URLSearchParams(Object.entries(filter).filter(([, v]) => v));
      if (history) q.set("history", "true");
      setRows(await api<Failure[]>(`/lab/failures?${q}`));
      if (!categories.length) setCategories(await api<string[]>("/lab/categories"));
    } catch (e) {
      setError(String((e as Error).message));
    }
  }, [filter, history, categories.length]);

  useEffect(() => {
    void load();
  }, [load]);

  async function promote(f: Failure) {
    setNotice(null);
    try {
      const r = await api<{ regression_id: string; file: string; created: boolean }>("/lab/regressions", {
        method: "POST",
        body: JSON.stringify({ eval_run_id: f.eval_run_id, scenario_id: f.scenario_id, run_index: f.run_index }),
      });
      setNotice(
        r.created
          ? `Promoted to ${r.regression_id} (evals/regressions/${r.file}). Commit the file to keep it.`
          : `${r.regression_id} was already a regression (${r.file}).`,
      );
      await load();
    } catch (e) {
      setError(String((e as Error).message));
    }
  }

  const select = (key: keyof typeof filter, options: string[], label: string) => (
    <label className="flex items-center gap-1 text-sm">
      {label}
      <select
        className="rounded border border-slate-300 px-1 py-0.5"
        value={filter[key]}
        onChange={(e) => setFilter({ ...filter, [key]: e.target.value })}
      >
        {options.map((o) => (
          <option key={o} value={o}>
            {o || "any"}
          </option>
        ))}
      </select>
    </label>
  );

  return (
    <Card title="Failures">
      <div className="flex flex-wrap items-center gap-3">
        {select("category", ["", ...categories], "Category")}
        {select("version", VERSIONS, "Version")}
        {select("label", LABELS, "Label")}
        <label className="flex items-center gap-1 text-sm" title="Also list aborted/failed runs and runs superseded by a re-score">
          <input type="checkbox" checked={history} onChange={(e) => setHistory(e.target.checked)} />
          Show history
        </label>
        <button className="rounded bg-slate-800 px-2 py-1 text-sm text-white" onClick={() => void load()}>
          Reload
        </button>
      </div>
      <ErrorLine error={error} />
      {notice && <p className="text-sm text-green-800">{notice}</p>}
      {rows && (
        <p className="text-sm text-slate-600">
          {rows.length} failed conversation{rows.length === 1 ? "" : "s"} in {history ? "all stored eval runs, including aborted and superseded" : "current eval runs (latest re-score of each run; aborted and failed runs hidden)"} (failed checks only).
        </p>
      )}
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-sm">
          <thead>
            <tr>
              {["Run", "Scenario", "Version", "Simulator", "Label", "Category", "Conversation", ""].map((h) => (
                <th key={h} className={th}>
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows?.map((f) => (
              <tr key={`${f.eval_run_id}-${f.scenario_id}-${f.run_index}`}>
                <td className={td}>
                  #{f.eval_run_id}
                  {f.run_status !== "completed" && (
                    <>
                      {" "}
                      <Chip tone="amber">{f.run_status}</Chip>
                    </>
                  )}
                </td>
                <td className={`${td} font-mono`}>
                  {f.scenario_id}
                  {f.run_index > 0 && <span className="text-slate-500"> [{f.run_index}]</span>}
                </td>
                <td className={td}>{f.agent_version}</td>
                <td className={td}>{f.simulator_version}</td>
                <td className={td}>
                  <Chip>{f.label}</Chip>
                </td>
                <td className={td}>
                  <div className="flex flex-wrap gap-1">
                    {f.categories.map((c) => (
                      <Chip key={c} tone={c === "critical_violation" ? "red" : "amber"}>
                        {c}
                      </Chip>
                    ))}
                  </div>
                </td>
                <td className={td}>
                  {f.conversation_id ? (
                    <Link className="text-blue-700 hover:underline" href={`/conversation/?id=${f.conversation_id}`}>
                      #{f.conversation_id}
                    </Link>
                  ) : (
                    <span className="text-slate-500">dial blocked</span>
                  )}
                </td>
                <td className={`${td} whitespace-nowrap`}>
                  {f.promoted ? (
                    <Chip tone="green">regression</Chip>
                  ) : (
                    <button className="text-blue-700 hover:underline" onClick={() => void promote(f)}>
                      Promote
                    </button>
                  )}{" "}
                  <Link className="text-blue-700 hover:underline" href={`/replay/?scenario=${encodeURIComponent(f.scenario_id)}`}>
                    Replay
                  </Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
