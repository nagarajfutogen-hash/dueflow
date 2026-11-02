"use client";

import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { Card, Chip, ErrorLine, td, th } from "@/components/ui";
import { api, type Detail } from "@/lib/api";

export default function ConversationPage() {
  return (
    <Suspense fallback={<p className="text-sm">Loading…</p>}>
      <ConversationView />
    </Suspense>
  );
}

function ConversationView() {
  const id = useSearchParams().get("id");
  const [d, setD] = useState<Detail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    api<Detail>(`/lab/conversations/${encodeURIComponent(id)}`)
      .then(setD)
      .catch((e: Error) => setError(e.message));
  }, [id]);

  if (!id) return <ErrorLine error="No conversation id in the address (?id=…)." />;
  if (error) return <ErrorLine error={error} />;
  if (!d) return <p className="text-sm">Loading…</p>;

  const c = d.conversation;
  const judge = d.scores.filter((s) => s.metric.startsWith("judge_") && s.metric !== "judge_error");
  const checks = d.scores.filter((s) => !s.metric.startsWith("judge_"));
  return (
    <div className="flex flex-col gap-4">
      <Card title={`Conversation #${c.id}`}>
        <div className="flex flex-wrap gap-2 text-sm">
          <Chip>{c.label}</Chip>
          <Chip>{c.channel}</Chip>
          <Chip>agent {c.agent_version}</Chip>
          {c.scenario_id && <Chip>{c.scenario_id}</Chip>}
          <Chip tone={c.outcome ? "slate" : "amber"}>outcome: {c.outcome ?? "none"}</Chip>
          {d.categories.map((cat) => (
            <Chip key={cat} tone={cat === "critical_violation" ? "red" : "amber"}>
              {cat}
            </Chip>
          ))}
        </div>
        <p className="text-sm text-slate-600">
          Simulated call time {c.sim_started_at} · {c.llm_calls} LLM calls · {c.prompt_tokens + c.completion_tokens}{" "}
          tokens
        </p>
      </Card>

      <Card title="Transcript">
        <ol className="flex flex-col gap-2">
          {d.turns.map((t) => (
            <li key={t.seq} className={t.speaker === "agent" ? "pr-12" : "pl-12 text-right"}>
              <div className="text-xs text-slate-500">
                {t.seq}. {t.speaker}
                {t.source && ` · ${t.source}`}
                {t.latency_ms?.interrupted && " · barge-in"}
              </div>
              <div className={`inline-block rounded px-3 py-1.5 text-sm ${t.speaker === "agent" ? "bg-blue-50" : "bg-slate-100"}`}>
                {t.text}
              </div>
            </li>
          ))}
        </ol>
      </Card>

      <Latency d={d} />

      <Card title="State transitions, tool calls and policy decisions (audit log)">
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-sm">
            <thead>
              <tr>
                {["#", "Type", "Rule", "Decision", "Reason"].map((h) => (
                  <th key={h} className={th}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {d.events.map((e) => (
                <tr key={e.id}>
                  <td className={td}>{e.id}</td>
                  <td className={td}>{e.type}</td>
                  <td className={`${td} font-mono`}>{e.rule ?? "–"}</td>
                  <td className={td}>
                    {e.decision && <Chip tone={e.decision === "blocked" ? "red" : "green"}>{e.decision}</Chip>}
                  </td>
                  <td className={td}>{e.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Checks and judge reasons">
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-sm">
            <thead>
              <tr>
                {["Run", "Metric", "Value", "Passed", "Evidence / reason"].map((h) => (
                  <th key={h} className={th}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[...checks, ...judge].map((s) => (
                <tr key={`${s.eval_run_id}-${s.run_index}-${s.metric}`}>
                  <td className={td}>#{s.eval_run_id}</td>
                  <td className={`${td} font-mono`}>{s.metric}</td>
                  <td className={td}>{s.value}</td>
                  <td className={td}>
                    {s.passed !== null && <Chip tone={s.passed ? "green" : "red"}>{s.passed ? "yes" : "no"}</Chip>}
                  </td>
                  <td className={`${td} max-w-md break-words font-mono text-xs`}>
                    {typeof s.evidence.reason === "string" ? s.evidence.reason : JSON.stringify(s.evidence)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

function Latency({ d }: { d: Detail }) {
  const data = d.turns
    .filter((t) => t.speaker === "agent" && typeof t.latency_ms?.perceived_ms === "number")
    .map((t) => ({
      turn: String(t.seq),
      perceived_ms: t.latency_ms?.perceived_ms as number,
      interrupted: Boolean(t.latency_ms?.interrupted),
    }));
  if (!data.length) return null;
  return (
    <Card title="Per-turn latency (voice, HUMAN TEST)">
      <p className="text-sm text-slate-600">
        Perceived latency per agent turn: end of the debtor&apos;s speech to the first TTS audio. Red bars: the debtor
        spoke over the agent (barge-in). n = {data.length}.
      </p>
      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="turn" label={{ value: "turn", position: "insideBottom", offset: -2 }} />
            <YAxis unit=" ms" width={70} />
            <Tooltip />
            <Bar dataKey="perceived_ms">
              {data.map((row) => (
                <Cell key={row.turn} fill={row.interrupted ? "#b91c1c" : "#1d4ed8"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}
