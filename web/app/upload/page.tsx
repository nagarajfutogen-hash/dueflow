"use client";

import Link from "next/link";
import { type ChangeEvent, useState } from "react";

import { PageTitle } from "@/components/admin";
import { api, ApiError } from "@/lib/api";

// Same columns, in the same order, as the server's import (app/debtors.py CSV_COLUMNS).
const COLUMNS: [string, boolean, string][] = [
  ["ref", true, "Your customer ID, e.g. C-1001 (must be new)"],
  ["full_name", true, "Name the agent asks for"],
  ["dob", true, "Date of birth, YYYY-MM-DD (used to confirm identity)"],
  ["balance_jpy", true, "Amount owed in whole yen, e.g. 120000"],
  ["timezone", false, "e.g. Asia/Tokyo (default)"],
  ["mobile", false, "Mobile number (at least one of mobile, home or work is required)"],
  ["home", false, "Home number"],
  ["work", false, "Work number"],
  ["stop_contact", false, "true = never call"],
  ["no_workplace_contact", false, "true = never call at work"],
  ["workplace_justification", false, "Why calling work is allowed"],
  ["open_dispute", false, "Only if the customer disputes the debt: the reason in words. Leave empty otherwise (any text opens a dispute)"],
  ["notes", false, "Background for the agent (max 500 characters)"],
  ["min_monthly_jpy", false, "Lowest monthly payment this customer may be offered, in yen (agent v5). Empty = no minimum"],
  ["max_months", false, "Most monthly payments this customer may be offered, 3-6 (agent v5). Empty = up to 6"],
];
const TEMPLATE =
  COLUMNS.map(([c]) => c).join(",") +
  "\nC-1001,Hana Example,1990-04-12,120000,Asia/Tokyo,090-0000-0001,,,false,false,,,New customer,15000,6\n";
const MAX_BYTES = 512_000;
const PREVIEW_ROWS = 8;

// Simple CSV split for the preview only (quoted commas supported); the server does the real parse.
function splitLine(line: string): string[] {
  const out: string[] = [];
  let cur = "";
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (quoted && line[i + 1] === '"') {
        cur += '"';
        i++;
      } else quoted = !quoted;
    } else if (ch === "," && !quoted) {
      out.push(cur);
      cur = "";
    } else cur += ch;
  }
  out.push(cur);
  return out;
}

type Result = { added: string[]; skipped: string[] };

export default function UploadPage() {
  const [name, setName] = useState<string | null>(null);
  const [text, setText] = useState<string | null>(null);
  const [rows, setRows] = useState<string[][]>([]);
  const [count, setCount] = useState(0);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<Result | null>(null);
  const [errors, setErrors] = useState<string[]>([]);

  async function pick(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    setResult(null);
    setErrors([]);
    if (!file) return;
    if (file.size > MAX_BYTES) {
      setErrors([`The file is too big (max ${MAX_BYTES / 1000} KB).`]);
      return;
    }
    const raw = (await file.text()).replace(/^﻿/, "");
    const lines = raw.split(/\r?\n/).filter((l) => l.trim());
    setName(file.name);
    setText(raw);
    setRows(lines.slice(0, PREVIEW_ROWS + 1).map(splitLine));
    setCount(Math.max(0, lines.length - 1));
  }

  async function upload() {
    if (!text) return;
    setBusy(true);
    setErrors([]);
    try {
      setResult(await api<Result>("/debtors/import", { method: "POST", body: text, headers: { "Content-Type": "text/csv" } }));
      setText(null);
    } catch (err) {
      const e = err as ApiError;
      setErrors(e.details?.length ? e.details : [e.message]);
    } finally {
      setBusy(false);
    }
  }

  const template = `data:text/csv;charset=utf-8,${encodeURIComponent(TEMPLATE)}`;

  return (
    <main className="flex flex-col gap-6">
      <PageTitle
        title="Upload customers"
        help="Add test customers from a CSV file. All rows must be valid, or nothing is added. Customers whose ID already exists are skipped."
      >
        <a href={template} download="dueflow-customers-template.csv" className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm hover:bg-slate-50">
          Download template
        </a>
      </PageTitle>

      <section className="flex flex-col gap-4 rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <label className="flex cursor-pointer flex-col items-center gap-2 rounded-lg border-2 border-dashed border-slate-300 px-4 py-8 text-center hover:border-slate-400">
          <span className="font-medium">{name ?? "Choose a CSV file"}</span>
          <span className="text-sm text-slate-500">
            {name ? `${count} customer row${count === 1 ? "" : "s"} found` : "UTF-8 CSV, up to 500 rows"}
          </span>
          <input type="file" accept=".csv,text/csv" className="hidden" onChange={pick} />
        </label>

        {rows.length > 1 && text && (
          <>
            <div className="overflow-x-auto">
              <p className="mb-2 text-sm text-slate-600">
                Preview{count > PREVIEW_ROWS ? ` (first ${PREVIEW_ROWS} of ${count})` : ""}
              </p>
              <table className="w-full text-xs">
                <thead className="bg-slate-50 text-left text-slate-600">
                  <tr>
                    {rows[0].map((h, i) => (
                      <th key={i} className="whitespace-nowrap px-2 py-1.5 font-medium">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.slice(1).map((r, i) => (
                    <tr key={i} className="border-t border-slate-100">
                      {r.map((v, j) => (
                        <td key={j} className="max-w-48 truncate whitespace-nowrap px-2 py-1.5">
                          {v}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <button
              onClick={upload}
              disabled={busy}
              className="self-start rounded-lg bg-slate-900 px-4 py-2 text-sm text-white disabled:opacity-50"
            >
              {busy ? "Importing…" : `Import ${count} customer${count === 1 ? "" : "s"}`}
            </button>
          </>
        )}

        {errors.length > 0 && (
          <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-800">
            <p className="font-medium">Nothing was imported. Fix these and try again:</p>
            <ul className="mt-1 list-disc pl-5">
              {errors.map((e, i) => (
                <li key={i}>{e}</li>
              ))}
            </ul>
          </div>
        )}

        {result && (
          <div className="rounded-lg border border-green-200 bg-green-50 p-3 text-sm text-green-900">
            <p className="font-medium">
              Added {result.added.length} customer{result.added.length === 1 ? "" : "s"}.
              {result.skipped.length > 0 && ` Skipped ${result.skipped.length} that already existed.`}
            </p>
            {result.added.length > 0 && <p className="mt-1">Added: {result.added.join(", ")}</p>}
            {result.skipped.length > 0 && <p className="mt-1">Already existed: {result.skipped.join(", ")}</p>}
            <p className="mt-2">
              They can now be called from the <a className="underline" href="/dev/">test call page</a>. See results in{" "}
              <Link className="underline" href="/calls/">All calls</Link>.
            </p>
          </div>
        )}
      </section>

      <section className="flex flex-col gap-2 rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <h2 className="font-semibold">Columns</h2>
        <table className="text-sm">
          <tbody>
            {COLUMNS.map(([c, required, help]) => (
              <tr key={c} className="border-t border-slate-100">
                <td className="whitespace-nowrap py-1.5 pr-4 font-mono text-xs">{c}</td>
                <td className="whitespace-nowrap py-1.5 pr-4 text-xs">
                  {required ? <span className="font-medium text-slate-900">Required</span> : <span className="text-slate-500">Optional</span>}
                </td>
                <td className="py-1.5 text-slate-600">{help}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="text-xs text-slate-500">
          The plan limits may only be stricter than the global rules, and only agent v5 uses them. There is no discount
          column: the full amount owed is always collected.
        </p>
      </section>
    </main>
  );
}
