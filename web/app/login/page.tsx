"use client";

import { useRouter } from "next/navigation";
import { type FormEvent, useState } from "react";

import { api, setToken } from "@/lib/api";

export default function LoginPage() {
  const router = useRouter();
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const { token } = await api<{ token: string }>("/auth/login", {
        method: "POST",
        body: JSON.stringify({ password }),
      });
      setToken(token);
      router.replace("/");
    } catch (err) {
      const msg = (err as Error).message;
      setError(msg.startsWith("401") ? "Wrong password." : msg.startsWith("429") ? "Too many tries. Wait a minute." : msg);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="flex min-h-screen items-center justify-center px-4">
      <form onSubmit={submit} className="flex w-full max-w-sm flex-col gap-4 rounded-lg border border-slate-200 bg-white p-6">
        <div>
          <h1 className="text-xl font-semibold">dueflow admin</h1>
          <p className="text-sm text-slate-600">Sign in to see calls and results.</p>
        </div>
        <label className="flex flex-col gap-1 text-sm">
          Password
          <input
            type="password"
            autoFocus
            autoComplete="current-password"
            className="rounded border border-slate-300 px-3 py-2"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
        </label>
        {error && <p className="text-sm text-red-700">{error}</p>}
        <button
          disabled={busy || !password}
          className="rounded bg-slate-900 px-3 py-2 text-white disabled:opacity-50"
        >
          {busy ? "Signing in…" : "Sign in"}
        </button>
      </form>
    </main>
  );
}
