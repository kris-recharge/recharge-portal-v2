'use client';

// app/(app)/app/page.tsx
// Auth-protected dashboard shell that will host the Streamlit app_v2 UI via iframe.

import { useEffect, useState } from 'react';
import { supabase } from '../../lib/supabaseClient';

// Prefer NEXT_PUBLIC_DASHBOARD, but fall back to the older *_URL var or a sane default.
const DASHBOARD_BASE =
  process.env.NEXT_PUBLIC_DASHBOARD ||
  process.env.NEXT_PUBLIC_DASHBOARD_URL ||
  'https://recharge-portal-v2-5k8n.onrender.com/';

const EMBED_TOKEN = process.env.NEXT_PUBLIC_EMBED_ACCESS_TOKEN || '';

type Session = Awaited<ReturnType<typeof supabase.auth.getSession>>['data']['session'];

export default function AppPage() {
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let mounted = true;

    // Initial session check
    supabase.auth.getSession().then(({ data }) => {
      if (!mounted) return;
      setSession(data.session ?? null);
      setLoading(false);

      if (!data.session) {
        // No active session: bounce to login
        window.location.href = '/';
      }
    });

    // Keep session in sync if auth state changes (e.g. sign out in another tab)
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, newSession) => {
      if (!mounted) return;
      setSession(newSession);
      setLoading(false);

      if (!newSession) {
        window.location.href = '/';
      }
    });

    return () => {
      mounted = false;
      subscription.unsubscribe();
    };
  }, []);

  const handleSignOut = async () => {
    await supabase.auth.signOut();
    window.location.href = '/';
  };

  if (loading) {
    return (
      <main className="min-h-screen flex items-center justify-center bg-slate-950 text-slate-100">
        <p className="text-sm text-slate-300">Checking your session…</p>
      </main>
    );
  }

  if (!session) {
    // Safety net; redirects above should usually handle this.
    return null;
  }

  const email = session.user?.email ?? 'ReCharge Portal User';

  // Build iframe src with embed token and email for the Streamlit sidebar.
  const params = new URLSearchParams();
  params.set('embedded', 'true');
  if (email) params.set('email', email);
  if (EMBED_TOKEN) params.set('token', EMBED_TOKEN);

  const iframeSrc = DASHBOARD_BASE.includes('?')
    ? `${DASHBOARD_BASE}&${params.toString()}`
    : `${DASHBOARD_BASE}?${params.toString()}`;

  return (
    <main className="h-screen flex flex-col bg-slate-950 text-slate-100 overflow-hidden">
      <header className="w-full border-b border-slate-800 px-6 py-3 flex items-center justify-between bg-slate-900/80 backdrop-blur">
        <div className="flex items-center gap-3">
          <div className="h-8 w-8 rounded-full bg-emerald-500/20 flex items-center justify-center text-xs font-bold tracking-tight">
            RA
          </div>
          <div className="flex flex-col">
            <h1 className="text-base font-semibold leading-tight">
              ReCharge Alaska – Operations Dashboard
            </h1>
            <p className="text-xs text-slate-400">
              Sessions · Status · Connectivity · Data Export
            </p>
          </div>
        </div>
        <div className="flex items-center gap-4 text-xs text-slate-400">
          <div className="text-right leading-tight">
            <div className="font-medium text-slate-100 truncate max-w-[14rem]">
              {email}
            </div>
            <div className="text-[0.7rem] text-slate-400">
              Protected area – access controlled by Supabase
            </div>
          </div>
          <button
            onClick={handleSignOut}
            className="px-3 py-1.5 rounded-xl border border-slate-600 text-xs font-medium hover:bg-slate-800 hover:border-slate-500 transition-colors"
          >
            Sign out
          </button>
        </div>
      </header>

      <section className="flex-1 min-h-0 flex flex-col">

        <div className="flex-1 min-h-0 bg-black/80">
          <iframe
            src={iframeSrc}
            title="ReCharge Alaska Dashboard"
            className="w-full h-full border-0"
            loading="lazy"
          />
        </div>
      </section>
    </main>
  );
}