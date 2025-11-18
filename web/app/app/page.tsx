// app/(app)/app/page.tsx
// Auth-protected dashboard shell that will host the Streamlit app_v2 UI via iframe.

const DASHBOARD_URL =
  process.env.NEXT_PUBLIC_DASHBOARD_URL ||
  "https://recharge-portal.onrender.com/app";

export default function AppPage() {
  return (
    <main className="min-h-screen flex flex-col bg-slate-950 text-slate-100">
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
        <div className="flex items-center gap-3 text-xs text-slate-400">
          <span className="hidden md:inline">
            Protected area – access controlled by Supabase
          </span>
        </div>
      </header>

      <section className="flex-1 flex flex-col">
        {/* Optional message row */}
        <div className="border-b border-slate-800 px-6 py-2 text-xs text-slate-400 bg-slate-900/60">
          If the dashboard below does not load, verify that the Render app is
          running at
          <code className="ml-1 font-mono text-[0.7rem]">
            {" "}
            {DASHBOARD_URL}
          </code>
          .
        </div>

        {/* Iframe container */}
        <div className="flex-1 bg-black/80">
          <iframe
            src={DASHBOARD_URL}
            title="ReCharge Alaska Dashboard"
            className="w-full h-full border-0"
            loading="lazy"
          />
        </div>
      </section>
    </main>
  );
}