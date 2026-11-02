"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { LabelBadge, PageTitle, ResultPill, when } from "@/components/admin";
import { CallbacksPanel } from "@/components/callbacks-panel";
import { ErrorLine } from "@/components/ui";
import { api, type CallRow } from "@/lib/api";

export default function LivePage() {
  const [rows, setRows] = useState<CallRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    const load = () =>
      api<CallRow[]>("/admin/calls?status=live&source=all")
        .then((v) => alive && setRows(v))
        .catch((e: Error) => alive && setError(e.message));
    load();
    const t = setInterval(load, 3_000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);

  return (
    <main className="flex flex-col gap-5">
      <PageTitle title="Live calls" help="Calls happening right now. Updates every 3 seconds; open one to follow the conversation." />
      <ErrorLine error={error} />
      <CallbacksPanel />
      {rows && rows.length === 0 && (
        <p className="rounded-lg border border-dashed border-slate-300 bg-white p-8 text-center text-slate-500">
          No call in progress.
        </p>
      )}
      <div className="grid gap-3 md:grid-cols-2">
        {rows?.map((c) => (
          <Link
            key={c.id}
            href={`/call/?id=${c.id}`}
            className="flex flex-col gap-2 rounded-lg border border-slate-200 bg-white p-4 hover:border-slate-400"
          >
            <div className="flex items-center justify-between gap-2">
              <span className="font-semibold">{c.customer}</span>
              <ResultPill result={c.result} status={c.status} words={c.result_words} />
            </div>
            <p className="text-sm text-slate-700">{c.summary}</p>
            <div className="flex items-center gap-2 text-xs text-slate-500">
              Started {when(c.started_at)} · {c.channel === "voice" ? "Voice" : "Chat"} <LabelBadge label={c.label} />
            </div>
          </Link>
        ))}
      </div>
    </main>
  );
}
