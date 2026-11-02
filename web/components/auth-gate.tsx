"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { type ReactNode, useEffect, useState } from "react";

import { Icon, type IconName } from "@/components/icons";
import { clearToken, getToken, LOGOUT_EVENT } from "@/lib/api";

const NAV: [string, string, IconName][] = [
  ["/", "Today", "home"],
  ["/live/", "Live calls", "live"],
  ["/calls/", "All calls", "calls"],
  ["/customers/", "Customers", "team"],
  ["/calendar/", "Calendar", "calendar"],
  ["/attention/", "Needs attention", "alert"],
  ["/upload/", "Upload customers", "upload"],
  ["/quality/", "Quality", "lab"],
];

// Every page except /login/ needs a token; a 401 from the API clears it and comes back here.
export function AuthGate({ children }: { children: ReactNode }) {
  const path = usePathname();
  const router = useRouter();
  const [ready, setReady] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const onLogin = path === "/login/" || path === "/login";

  useEffect(() => {
    if (!onLogin && !getToken()) router.replace("/login/");
    else setReady(true);
    const out = () => router.replace("/login/");
    window.addEventListener(LOGOUT_EVENT, out);
    return () => window.removeEventListener(LOGOUT_EVENT, out);
  }, [onLogin, router]);

  useEffect(() => setMenuOpen(false), [path]);

  if (onLogin) return <>{children}</>;
  if (!ready) return null;

  const isActive = (href: string) =>
    href === "/" ? path === "/" : path.startsWith(href.replace(/\/$/, "")) ||
      (href === "/calls/" && (path === "/call" || path.startsWith("/call/"))) ||
      (href === "/customers/" && path.startsWith("/customer")) ||
      (href === "/quality/" &&
        (path.startsWith("/lab") || path.startsWith("/conversation") || path.startsWith("/replay")));

  return (
    <div className="min-h-screen md:flex">
      <aside
        className={`${menuOpen ? "flex" : "hidden"} fixed inset-0 z-20 flex-col bg-slate-900 text-slate-200 md:sticky md:top-0 md:flex md:h-screen md:w-60 md:shrink-0`}
      >
        <div className="flex items-center justify-between px-5 py-5">
          <div>
            <div className="text-lg font-semibold text-white">dueflow</div>
            <div className="text-xs text-slate-400">Collections admin</div>
          </div>
          <button className="md:hidden" onClick={() => setMenuOpen(false)} aria-label="Close menu">
            ✕
          </button>
        </div>
        <nav className="flex flex-1 flex-col gap-1 px-3">
          {NAV.map(([href, text, icon]) => (
            <Link
              key={href}
              href={href}
              className={`flex items-center gap-3 rounded-lg px-3 py-2 text-sm ${
                isActive(href) ? "bg-white/10 font-medium text-white" : "text-slate-300 hover:bg-white/5 hover:text-white"
              }`}
            >
              <Icon name={icon} />
              {text}
            </Link>
          ))}
        </nav>
        <button
          className="m-3 flex items-center gap-3 rounded-lg px-3 py-2 text-sm text-slate-400 hover:bg-white/5 hover:text-white"
          onClick={() => {
            clearToken();
            router.replace("/login/");
          }}
        >
          <Icon name="logout" />
          Log out
        </button>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center gap-3 border-b border-amber-200 bg-amber-50 px-4 py-1.5 text-xs text-amber-900 md:px-8">
          <button className="md:hidden" onClick={() => setMenuOpen(true)} aria-label="Open menu">
            <Icon name="menu" />
          </button>
          Test data only: SYNTHETIC (simulated customers) and HUMAN TEST (people role-playing). No real
          customers or collection results yet.
        </div>
        <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-4 py-6 md:px-8">{children}</div>
      </div>
    </div>
  );
}
